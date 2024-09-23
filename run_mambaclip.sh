num_gpus=1
num_nodes=1
global_batch_size=$((2**8))
num_seen_samples=$((30*1000*global_batch_size))
exp_name="mambaCLIP_datacompdr12m_s30m_single_gpu_$(date +%Y-%m-%d_%H-%M-%S)"
num_checkpoints=20

# TODO:
# 1. 모델 사이즈별 config 파일 만들고. 모델명으로 구분하도록 수정(보류, factory 파일에 구현된 상태)
# 2. batch size 2*13 = 8192로 수정(이게 안되기 때문에 grad_accum로 조절)
# 3. num_seen_samples 13B로 수정

# distill-loss-weights
# MobileCLIP-S1 -> M4CLIP-T: 0.0 1.0 
# MobileCLIP-S2 -> M4CLIP-S: 0.1 0.9
# MobileCLIP-B -> M4CLIP-B: 0.25 0.75

data="DataCompDR/{00000000..00000010}.tar"


model="MobileCLIP-S1"

CUDA_VISIBLE_DEVICES=0 python -m src.training.main \
    --save-frequency 1 \
    --local-loss \
    --accum-freq 1 \
    --gather-with-grad \
    --train-data "$data" \
    --warmup 1000 \
    --dataset-type webdataset \
    --train-num-samples $((num_seen_samples / num_checkpoints)) \
    --precision amp \
    --workers 16 \
    --model "$model"  \
    --batch-size $global_batch_size \
    --epochs $num_checkpoints \
    --lr 1.e-3 \
    --name $exp_name \
    --seed 0 \
    --log-every-n-steps 1 \
    --beta2 0.95 \
    --report-to wandb \
    --wd 0.2 \
    --dataset-resampled \
    --save-most-recent \
    --grad-clip-norm 1.0 \
    --imagenet-val "imagenet_validation/val" \
    --zeroshot-frequency 1 \
    --wandb-project-name "$model" \
    --dataset-reinforcement \
    --dataset-reinforcement-config datacompdr12m.json \
    --distill-logit-scale 100 \
    --distill-loss-weights 0.0 1.0 \
    --distill-teacher-dimension 768 768 \
    --distill-average-after-softmax \
