num_gpus=1
num_nodes=1
global_batch_size=$((2**11))
num_seen_samples=$((30*1000*global_batch_size))
exp_name="mambaCLIP_datacompdr12m_s30m_single_gpu_$(date +%Y-%m-%d_%H-%M-%S)"
num_checkpoints=20  

# grad_checkpointing

# TODO:
# 1. 모델 사이즈별 config 파일 만들고. 모델명으로 구분하도록 수정
# 2. batch size 2*13 = 8192로 수정
# 3. num_seen_samples 13B로 수정

# HuggingFace 데이터셋 URL로 변경
data="pipe:curl -L -s -H 'Authorization: Bearer ${HUGGINGFACE_TOKEN}' https://huggingface.co/datasets/apple/DataCompDR-12M/resolve/main/{00000000..00000000}.tar"

CUDA_VISIBLE_DEVICES=0 python -m src.training.main \
    --save-frequency 1 \
    --local-loss \
    --accum-freq 4 \
    --gather-with-grad \
    --train-data "$data" \
    --train-num-samples 7 \
    --warmup 1000 \
    --dataset-type datacomp \
    --precision amp \
    --workers 1 \
    --model MambaCLIP  \
    --batch-size $global_batch_size \
    --epochs $num_checkpoints \
    --lr 1.e-3 \
    --name $exp_name \
    --seed 0 \
    --log-every-n-steps 1 \
    --beta2 0.95 \
    --wd 0.2 \
    --dataset-resampled \
    --save-most-recent \
    --grad-clip-norm 1.0 \
    --imagenet-val "imagenet_validation/val" \
    --zeroshot-frequency 1 \
    --wandb-project-name mamba2clip \
    --dataset-reinforcement \
    --dataset-reinforcement-config datacompdr12m.json \
    --distill-logit-scale 100 \
    --distill-loss-weights 0.0 1.0 \
    --distill-teacher-dimension 768 768 \
    --distill-average-after-softmax \
    --mamba-d-model 512 \
    --mamba-n-layer 12 \
