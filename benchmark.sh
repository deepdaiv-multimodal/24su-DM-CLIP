# python -m benchmark.cli eval \
#     --model_type "mobile_clip" \
#     --model "MobileCLIP-S1" \
#     --dataset "benchmark/webdatasets.txt" \
#     --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
#     --output "results/{model}.json" \
#     --pretrained "checkpoints/mobilemclip_s1_6m_0_1.pt" \
#     --image_encoder_id "nvidia/MambaVision-B-1K" \
#     --batch_size 1024 \

# python -m benchmark.cli eval \
#     --model_type "mobile_clip" \
#     --model "MobileCLIP-S1" \
#     --dataset "benchmark/webdatasets.txt" \
#     --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
#     --output "results/{model}.json" \
#     --pretrained "checkpoints/mobilemclip_s1_6m_1_0.pt" \
#     --image_encoder_id "nvidia/MambaVision-B-1K" \
#     --batch_size 1024 \

# python -m benchmark.cli eval \
#     --model_type "mobile_clip" \
#     --model "MobileCLIP-S1" \
#     --dataset "benchmark/webdatasets.txt" \
#     --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
#     --output "results/{model}.json" \
#     --pretrained "checkpoints/mobilemclip_s1_6m_025_075.pt" \
#     --image_encoder_id "nvidia/MambaVision-B-1K" \
#     --batch_size 1024 \

python -m benchmark.cli eval \
    --model_type "mobile_clip" \
    --model "MobileCLIP-S1" \
    --dataset "benchmark/webdatasets.txt" \
    --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
    --output "results/{model}.json" \
    --pretrained "checkpoints/mobilemclip_s1_6m_0_1.pt" \
    --image_encoder_id "nvidia/MambaVision-B-1K" \
    --batch_size 2048 \

# python -m benchmark.cli eval \
#     --model_type "mobile_clip" \
#     --model "MobileCLIP-S1" \
#     --dataset "benchmark/webdatasets.txt" \
#     --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
#     --output "results/{model}.json" \
#     --pretrained "checkpoints/mobileclip_s1_6m_1_0.pt" \
#     --batch_size 1024 \

# python -m benchmark.cli eval \
#     --model_type "mobile_clip" \
#     --model "MobileCLIP-S1" \
#     --dataset "benchmark/webdatasets.txt" \
#     --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
#     --output "results/{model}.json" \
#     --pretrained "checkpoints/mobileclip_s1_6m_025_075.pt" \
#     --batch_size 1024 \

# python -m benchmark.cli eval \
#     --model_type "mobile_clip" \
#     --model "MobileCLIP-S1" \
#     --dataset "benchmark/webdatasets.txt" \
#     --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
#     --output "results/{model}.json" \
#     --pretrained "checkpoints/mobileclip_s1_6m_075_025.pt" \
#     --batch_size 1024 \