
python -m benchmark.cli eval \
    --model_type "mobile_clip" \
    --model "MobileCLIP-S2" \
    --dataset "benchmark/webdatasets.txt" \
    --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
    --output "results/{model}.json" \
    --pretrained "checkpoints/mobilemlit_s2.pt" \
    --image_encoder_id "nvidia/MambaVision-L-1K" \