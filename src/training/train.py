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

    data['train'].set_epoch(epoch)

    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    # Gradient Accumulation 용 리스트 및 딕셔너리 초기화
    accum_images, accum_texts, accum_teacher_images, accum_teacher_texts, accum_syn_texts, accum_features = [], [], [], [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts, teacher_images, teacher_texts = batch[:4]
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        teacher_images = teacher_images.to(device=device, non_blocking=True)
        teacher_texts = teacher_texts.to(device=device, non_blocking=True)

        if args.dataset_reinforcement and not args.dataset_reinforcement_mix_synthetic:
            syn_texts = batch[4].to(device=device, non_blocking=True)
            teacher_syn_texts = batch[5].to(device=device, non_blocking=True)
            texts = torch.cat([texts, syn_texts[:, :texts.shape[-1]]], dim=0)

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
                    batch_size = images.shape[0]
                    model_out.update({
                        'dist_image_features': teacher_images,
                        'dist_text_features': teacher_texts,
                    })
                    if not args.dataset_reinforcement_mix_synthetic:
                        model_out.update({
                            "text_features": model_out["text_features"][:batch_size],
                            "syn_text_features": model_out["text_features"][batch_size:],
                            'dist_syn_text_features': teacher_syn_texts
                        })

                losses = loss(**model_out, output_dict=True)
                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)

        else:
            # Gradient Accumulation
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)
                accum_teacher_images.append(teacher_images)
                accum_teacher_texts.append(teacher_texts)
                if args.dataset_reinforcement and not args.dataset_reinforcement_mix_synthetic:
                    accum_syn_texts.append(teacher_syn_texts)

            if ((i + 1) % args.accum_freq) > 0:
                print("accum_freq: ", args.accum_freq)
                continue

            logit_scale_scalar_accum = 0.0  # logit_scale 누적 변수 추가

            for j in range(args.accum_freq):
                print("j: ", j)
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():  # Student 모델 출력값 사용
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    if args.dataset_reinforcement:
                        batch_size = images.shape[0]
                        inputs.update({
                            'dist_image_features': accum_teacher_images[j].detach(),  # detach 추가
                            'dist_text_features': accum_teacher_texts[j].detach(),  # detach 추가
                        })
                        if not args.dataset_reinforcement_mix_synthetic:
                            inputs.update({
                                "text_features": inputs["text_features"][:batch_size],
                                "syn_text_features": inputs["text_features"][batch_size:],
                                'dist_syn_text_features': accum_syn_texts[j].detach()  # detach 추가
                            })

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                    backward(total_loss / args.accum_freq, scaler)  # Gradient 누적

                    # losses_m 업데이트
                    for key, val in losses.items():
                        if key not in losses_m:
                            losses_m[key] = AverageMeter()
                        losses_m[key].update(val.item(), batch_size)

                    logit_scale_scalar_accum += logit_scale.item()  # logit_scale 누적

            logit_scale_scalar = logit_scale_scalar_accum / args.accum_freq

            if ((i + 1) % args.accum_freq) == 0:
                batch_time_m.update(time.time() - end)
                end = time.time()
                batch_count = i_accum + 1

                # 매 accum_freq 스텝마다 로그 출력
                print("batch_count: ", batch_count)
                batch_size = len(images)
                num_samples = batch_count * batch_size * args.accum_freq * args.world_size
                samples_per_epoch = dataloader.num_samples
                percent_complete = 100.0 * batch_count / num_batches_per_epoch

                # losses 값을 accum_freq로 나누어 평균 loss 계산
                for key, val in losses_m.items():
                    losses[key] = val.avg

                logit_scale_scalar = logit_scale_scalar # .item() 제거
                loss_log = " ".join(
                    [
                        f"{loss_name.capitalize()}: {loss_m.avg:#.5g}" # loss_m.val -> loss_m.avg
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

                log_data = {
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                    "samples_per_second": samples_per_second,
                    "samples_per_second_per_gpu": samples_per_second_per_gpu,
                    "scale": logit_scale_scalar,
                    "lr": optimizer.param_groups[0]["lr"]
                }
                log_data.update({name: val for name, val in losses.items()}) # 모든 loss 추가

                log_data = {"train/" + name: val for name, val in log_data.items()}

                if tb_writer is not None:
                    for name, val in log_data.items():
                        tb_writer.add_scalar(name, val, step)

                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    log_data['step'] = step  # for backwards compatibility
                    wandb.log(log_data, step=step)

                batch_time_m.reset()
                data_time_m.reset()

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

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_teacher_images, accum_teacher_texts, accum_syn_texts, accum_features = [], [], [], [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        print("model: ", model)
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        

def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    # if not is_master(args):
    #     return metrics
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

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
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
