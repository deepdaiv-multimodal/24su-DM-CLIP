# batch_size = 8192
# 현재는 메모리 이슈로 2048로 학습 중
global_batch_size=$((2**11))
grad_accum=1

# 100M
num_seen_samples=$((50*1000*global_batch_size))
num_checkpoints=20

# distill-loss-weights
# MobileCLIP-S1 -> MobileMLiT-S1: 0.25 0.75
# MobileCLIP-S2 -> MobileMLiT-S2: 0.25 0.75
# MobileCLIP-B -> MobileMLiT-B: 0.25 0.75

data="DataCompDR-5M/{00000000..00000500}.tar"
model="MobileCLIP-S1"

if [ "$model" = "MobileCLIP-S1" ]; then
    wandb_project_name="MobileMLiT_S1"
    exp_name="MobileMLiT_S1_$(date +%Y-%m-%d_%H-%M-%S)"
    pretrained="checkpoints/mobileclip_s1.pt"
elif [ "$model" = "MobileCLIP-S2" ]; then
    wandb_project_name="MobileMLiT_S2"
    exp_name="MobileMLiT_S2_$(date +%Y-%m-%d_%H-%M-%S)"
    pretrained="checkpoints/mobileclip_s2.pt"
elif [ "$model" = "MobileCLIP-B" ]; then
    wandb_project_name="MobileMLiT_B"
    exp_name="MobileMLiT_B_$(date +%Y-%m-%d_%H-%M-%S)"
    pretrained="checkpoints/mobileclip_b.pt"
else
    echo "Invalid model name"
    exit 1
fi


CUDA_VISIBLE_DEVICES=0 python -m src.training.main \
    --save-frequency 1 \
    --pretrained "$pretrained" \
    --local-loss \
    --accum-freq "$grad_accum" \
    --gather-with-grad \
    --train-data "$data" \
    --report-to wandb \
    --warmup 1000 \
    --dataset-type webdataset \
    --train-num-samples $((num_seen_samples / num_checkpoints)) \
    --precision amp \
    --workers 8 \
    --model "$model"  \
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
    --wandb-project-name "$wandb_project_name" \
    --dataset-reinforcement \
    --dataset-reinforcement-config datacompdr12m.json \
    --distill-logit-scale 100 \
    --distill-loss-weights 0.25 0.75 \
    --distill-teacher-dimension 768 768 \
    --distill-average-after-softmax \