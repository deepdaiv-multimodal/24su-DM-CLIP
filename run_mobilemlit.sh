# batch_size = 8192
global_batch_size=$((2**11))
grad_accum=1

# 50M
num_seen_samples=$((25*1000*global_batch_size)) # global_batch_size
num_checkpoints=20

data="DataCompDR-5M/{00000000..00000500}.tar"
model="MobileCLIP-S2"

if [ "$model" = "MobileCLIP-S1" ]; then
    wandb_project_name="MobileMLiT_S1"
    exp_name="MobileMLiT_S1_$(date +%Y-%m-%d_%H-%M-%S)"
    pretrained="checkpoints/mobileclip_s1.pt"
    image_encoder_id="nvidia/MambaVision-B-1K"
elif [ "$model" = "MobileCLIP-S2" ]; then
    wandb_project_name="MobileMLiT_S2"
    exp_name="MobileMLiT_S2_$(date +%Y-%m-%d_%H-%M-%S)"
    pretrained="checkpoints/mobileclip_s2.pt"
    image_encoder_id="nvidia/MambaVision-L-1K"
else
    echo "Invalid model name"
    exit 1
fi

#    --pretrained "$pretrained" \
#     --report-to wandb \
#     --local-loss \
#     --gather-with-grad \

CUDA_VISIBLE_DEVICES=0 python -m src.training.main \
    --save-frequency 1 \
    --accum-freq "$grad_accum" \
    --train-data "$data" \
    --warmup 1000 \
    --dataset-type webdataset \
    --train-num-samples $((num_seen_samples / num_checkpoints)) \
    --precision amp_bfloat16 \
    --workers 4 \
    --report-to wandb \
    --local-loss \
    --gather-with-grad \
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
    --image-encoder-id "$image_encoder_id"