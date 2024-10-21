# batch_size = 8192
global_batch_size=$((2**8))
grad_accum=1

# 0.64B
num_seen_samples=$((global_batch_size)) # global_batch_size
num_checkpoints=20

data="DataCompDR-5M/{00000000..00000500}.tar"
model="MobileCLIP-S1"

if [ "$model" = "MobileCLIP-S1" ]; then
    wandb_project_name="MobileCLIP_M1"
    pretrained="checkpoints/mobileclip_s1.pt"
    exp_name="MCi_test_$(date +%Y-%m-%d_%H-%M-%S)"
    image_encoder_id="nvidia/MambaVision-B-1K"
else
    echo "Invalid model name"
    exit 1
fi

#    --pretrained "$pretrained" \
#     --report-to wandb \
#     --local-loss \
#     --gather-with-grad \
#     --image-encoder-id "$image_encoder_id" \ apple/mobileclip_s2_timm
#        --pretrained "$pretrained" \
#         --image-encoder-id "$image_encoder_id" \

CUDA_VISIBLE_DEVICES=0 python -m src.training.main \
    --save-frequency 1 \
    --accum-freq "$grad_accum" \
    --report-to wandb \
    --train-data "$data" \
    --warmup 1000 \
    --dataset-type webdataset \
    --train-num-samples $((num_seen_samples / num_checkpoints)) \
    --precision amp_bfloat16 \
    --workers 8 \
    --local-loss \
    --gather-with-grad \
    --model "$model"  \
    --batch-size $global_batch_size \
    --epochs $num_checkpoints \
    --lr 1.e-4 \
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
    --distill-loss-weights 0.75 0.25 \
    --distill-teacher-dimension 768 768 \
    --distill-average-after-softmax \
    --image-encoder-id "$image_encoder_id"