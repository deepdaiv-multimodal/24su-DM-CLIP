
#     --image_encoder_id "nvidia/MambaVision-T-1K" \
    # --image_encoder_id "nvidia/MambaVision-B-1K" \

# python -m benchmark.cli eval \
#     --model_type "mobile_clip" \
#     --model "MobileCLIP-S1" \
#     --dataset "benchmark/webdatasets.txt" \
#     --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
#     --output "results/{model}.json" \
#     --pretrained "checkpoints/mobilemclip_s1_6m.pt" \
#     --image_encoder_id "nvidia/MambaVision-B-1K" \
#     --batch_size 1024 \

python -m benchmark.cli eval \
    --model_type "mobile_clip" \
    --model "MobileCLIP-S1" \
    --dataset "benchmark/webdatasets.txt" \
    --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
    --output "results/{model}.json" \
    --pretrained "checkpoints/mobileclip_s1_6m.pt" \
    --batch_size 1024 \


# python -m benchmark.cli eval \
#     --model_type "mobile_clip" \
#     --model "MobileCLIP-S1" \
#     --dataset "benchmark/webdatasets.txt" \
#     --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
#     --output "results/{model}.json" \
#     --pretrained "checkpoints/mobileclip_s1.pt" \
#     --batch_size 512 \
