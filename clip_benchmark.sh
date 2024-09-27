clip_benchmark eval \
    --model_type "mobile_clip" \
    --model "mobileclip_s2" \
    --dataset "webdatasets.txt" \
    --batch_size 128\
    --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
    --output "/root/code/harim/DeepDaiv/M4-CLIP/benchmark/output/benchmark_mobileclip_{dataset}_{pretrained}_{model}_{language}_{task}.json"

# model "mobileclip_s1"