num_gpus=1
num_nodes=1
global_batch_size=$((2**11))
num_seen_samples=$((30*1000*global_batch_size))
exp_name="mambaCLIP_datacompdr12m_s30m_single_gpu_$(date +%Y-%m-%d_%H-%M-%S)"
num_checkpoints=20  


data="DataCompDR-12M/merged_shards/{00000000..00000000}.tar"    
#data="https://huggingface.co/datasets/apple/DataCompDR-12M-bf16/resolve/main/{00000000..00000000}.tar"
#     --report-to wandb \

CUDA_VISIBLE_DEVICES=0 python -m src.training.main \
    --save-frequency 1 \
    --local-loss \
    --accum-freq 4 \
    --gather-with-grad \
    --grad-checkpointing \
    --train-data "$data" \
    --train-num-samples 7 \
    --warmup 1000 \
    --dataset-type webdataset \
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
    --mamba-d-model 480 \
    --mamba-n-layer 24 \
