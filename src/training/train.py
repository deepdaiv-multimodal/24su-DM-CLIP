import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_syn_texts, accum_features = [], [], [], {}
        logit_scale = None

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch[:2]
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        # 데이터 증강을 위한 합성 텍스트를 사용
        if args.dataset_reinforcement and not args.dataset_reinforcement_mix_synthetic:
            syn_texts = batch[4].to(device=device, non_blocking=True)
            original_texts = texts
            texts = torch.cat([texts, syn_texts[:, :texts.shape[-1]]], dim=0)

        batch_size = images.shape[0]  # 배치 크기
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                if args.dataset_reinforcement:
                    model_out.update({
                        'dist_image_features': batch[2].to(device=device, non_blocking=True),
                        'dist_text_features': batch[3].to(device=device, non_blocking=True),
                    })
                    if not args.dataset_reinforcement_mix_synthetic:
                        model_out.update({
                            "text_features": model_out["text_features"][:batch_size],  # 원본 텍스트만 사용
                            "syn_text_features": model_out["text_features"][batch_size:],  # 합성 텍스트 사용
                            'dist_syn_text_features': batch[5].to(device=device, non_blocking=True)
                        })
                losses = loss(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    if args.distill:
                        with torch.no_grad():
                            dist_model_out = dist_model(images, texts)
                        model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                    if args.dataset_reinforcement:
                        model_out.update({
                            'dist_image_features': batch[2].to(device=device, non_blocking=True),
                            'dist_text_features': batch[3].to(device=device, non_blocking=True),
                        })
                        if not args.dataset_reinforcement_mix_synthetic:
                            model_out.update({
                                "text_features": model_out["text_features"][:batch_size],  # 원본 텍스트만 사용
                                "syn_text_features": model_out["text_features"][batch_size:],  # 합성 텍스트 사용
                                'dist_syn_text_features': batch[5].to(device=device, non_blocking=True)
                            })

                    for key, val in model_out.items():
                        if key not in accum_features:
                            accum_features[key] = []
                        accum_features[key].append(val)
                            
            # Cache the images and texts for the backward pass.
            accum_images.append(images.cpu())
            accum_texts.append(original_texts.cpu())
            accum_syn_texts.append(syn_texts.cpu())
            for key, val in model_out.items():
                if key not in accum_features:
                    accum_features[key] = []
                accum_features[key].append(val.cpu())

            if ((i + 1) % args.accum_freq) > 0:
                continue

            # Re-do the forward pass for the last accum_freq batches and use the cached features as negatives.
            optimizer.zero_grad()
            with autocast():
                total_loss_accum = 0.0
                for j in range(args.accum_freq):
                    images = accum_images[j].to(device)
                    texts = accum_texts[j].to(device)
                    syn_texts = accum_syn_texts[j].to(device)

                    texts = torch.cat([texts, syn_texts[:, :texts.shape[-1]]], dim=0)

                    model_out = model(images, texts)
                    inputs_no_accum = {}
                    if logit_scale is None:
                        logit_scale = model_out["logit_scale"]

                    inputs_no_accum["logit_scale"] = logit_scale

                    if args.distill:
                        with torch.no_grad():
                            dist_model_out = dist_model(images, texts)
                        model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})

                    if args.dataset_reinforcement and not args.dataset_reinforcement_mix_synthetic:
                        model_out.update({
                            "text_features": model_out["text_features"][:batch_size],
                            "syn_text_features": model_out["text_features"][batch_size:],
                        })

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = [tensor.to(device) for tensor in accum_features[key]]  # GPU로 다시 이동
                        if key == "dist_image_features" or key == "dist_text_features" or key == "dist_syn_text_features":
                            inputs[key] = torch.cat(accumulated)
                        else:
                            inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    total_loss_accum += total_loss

                losses["loss"] = total_loss_accum
                scaler.scale(total_loss_accum / args.accum_freq).backward()
                    
        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()

        if ((i + 1) % args.accum_freq) == 0:
            # reset the accumulated features
            accum_images, accum_texts, accum_syn_texts, accum_features = [], [], [], {}
            logit_scale = None
    # end for

def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        image_encoder_time = 0.0
        text_encoder_time = 0.0
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    # Start measuring image encoder time
                    start_time = time.time()
                    model_out = model(images, texts)
                    image_encoder_time += time.time() - start_time

                    image_features = model_out["image_features"]

                    # Start measuring text encoder time
                    start_time = time.time()
                    text_features = model_out["text_features"]
                    text_encoder_time += time.time() - start_time

                    logit_scale = model_out["logit_scale"]

                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()

                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            # Calculate FPS for image and text encoders
            image_encoder_fps = num_samples / image_encoder_time if image_encoder_time > 0 else 0
            text_encoder_fps = num_samples / text_encoder_time if text_encoder_time > 0 else 0

            # Log FPS
            logging.info(f"Image Encoder FPS: {image_encoder_fps:.2f}")
            logging.info(f"Text Encoder FPS: {text_encoder_fps:.2f}")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

            # Add FPS to metrics
            metrics.update({"image_encoder_fps": image_encoder_fps, "text_encoder_fps": text_encoder_fps})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
