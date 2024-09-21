import tarfile
import os
import json
from collections import defaultdict
import braceexpand
import requests
from io import BytesIO
from PIL import Image
import shutil  
import webdataset as wds
from multiprocessing import Pool
import torch
import gzip
from tqdm import tqdm


def merge_and_download_images(shard_idx, output_dir):
    """
    두 개의 DataCompDR-12M 데이터셋 샤드를 병합하고, 이미지를 다운로드하여 새로운 .tar 파일로 저장합니다.

    Args:
        shard_idx: 샤드 파일 인덱스 (0부터 시작)
        output_dir: 새로운 .tar 파일을 저장할 디렉토리 경로
    """
    os.makedirs(output_dir, exist_ok=True)

    # 데이터셋 URL (스트리밍 방식)
    BF16_URL = f"https://huggingface.co/datasets/apple/DataCompDR-12M-bf16/resolve/main/0000000{shard_idx}.tar"
    DATA_COMP12M_URL = f"https://huggingface.co/datasets/mlfoundations/DataComp-12M/resolve/main/0000000{shard_idx}.tar"

    # 두 데이터셋을 WebDataset 객체로 생성
    bf16_dataset = wds.WebDataset(BF16_URL)
    datacomp12m_dataset = wds.WebDataset(DATA_COMP12M_URL)

    # 파이프라인 정의
    bf16_pipeline = [
        wds.decode("pilrgba", handler=wds.handlers.warn_and_continue),
        wds.rename(url="url.txt", syn="syn.json", paug="paug.json", pth="pth.gz", json="json"),
        wds.to_tuple("url", "syn", "paug", "pth", "json"),   
    ]
    datacomp12m_pipeline = [
        wds.decode("pilrgba", handler=wds.handlers.warn_and_continue),
        wds.rename(json="json", txt="txt"),
        wds.to_tuple("json", "txt"),
    ]
    bf16_dataset = bf16_dataset.compose(*bf16_pipeline)
    datacomp12m_dataset = datacomp12m_dataset.compose(*datacomp12m_pipeline)

    # 샘플 정보 저장
    samples = defaultdict(dict)

    # 임시 폴더 생성
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # bf16 데이터셋에서 샘플 정보 추출
    for sample in bf16_dataset:
        # json 파일에 uid가 포함
        uid = sample[4]["uid"]
        samples[uid]["url"] = sample[0]
        samples[uid]["syn"] = sample[1]
        samples[uid]["paug"] = sample[2]
        samples[uid]["pth"] = sample[3]
        samples[uid]["json"] = sample[4]

    # DataComp12M 데이터셋에서 txt 파일 정보 추출 및 병합
    for sample in datacomp12m_dataset:
        uid = sample[0]["uid"]
        samples[uid]["txt"] = sample[1]

    # 성공적으로 다운로드된 샘플 개수
    success_count = 0
    total_samples = 0

    # 이미지 다운로드 및 저장 (tqdm 사용)
    for uid, data in samples.items():
        url = data["url"]
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()  # HTTP 에러 발생 시 예외 발생
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image_name = f"{uid}.jpg"
            image_path = os.path.join(temp_dir, image_name)
            image.save(image_path, "JPEG")

            # 나머지 파일 저장 (파일 경로 수정)
            with open(os.path.join(temp_dir, f"{uid}.syn.json"), "w") as f:
                json.dump(data["syn"], f)
            with open(os.path.join(temp_dir, f"{uid}.paug.json"), "w") as f:
                json.dump(data["paug"], f)

            pth_path = os.path.join(temp_dir, f"{uid}.pth.gz")
            with gzip.open(pth_path, "wb") as f:
                torch.save(data["pth"], f)

            with open(os.path.join(temp_dir, f"{uid}.json"), "w") as f:
                json.dump(data["json"], f)

            with open(os.path.join(temp_dir, f"{uid}.txt"), "w") as f:
                f.write(data["txt"])

            success_count += 1
            total_samples += 1

        except Exception as e:
            # print(f"Error downloading or saving image from {url}: {e}")
            # 이미지 다운로드 실패 시 해당 샘플 관련 파일 제거
            for ext in ['.jpg', '.txt', '.json', '.syn.json', '.paug.json', '.pth.gz']:
                file_path = os.path.join(temp_dir, uid + ext)
                if os.path.exists(file_path):
                    os.remove(file_path)
            total_samples += 1
            continue

    # 임시 폴더를 .tar 파일로 압축
    output_shard_path = os.path.join(output_dir, f"0000000{shard_idx}.tar")
    with tarfile.open(output_shard_path, "w") as new_tar:
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            new_tar.add(file_path, arcname=filename)

    print(f"Filtered and repacked shard (with {len(os.listdir(temp_dir))} samples) saved to: {output_shard_path}")
    print(f"Shard {shard_idx}: Success rate: {success_count / total_samples * 100:.2f}%")  # 성공률 출력

    # 임시 폴더 삭제
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # 샤드 파일 개수 설정 (0부터 시작)
    start_shard_idx = 1
    end_shard_idx = 1

    # 새로운 .tar 파일 저장 디렉토리
    output_dir = "DataCompDR-12M/merged_shards"

    # 병렬 처리를 위한 프로세스 풀 생성 (CPU 코어 개수에 맞춰 조정 가능)
    num_processes = os.cpu_count()
    print(f"Using {num_processes} processes for parallel processing.")
    with Pool(processes=num_processes) as pool:
        # 각 샤드 파일을 병렬로 처리
        pool.starmap(merge_and_download_images, [(i, output_dir) for i in range(start_shard_idx, end_shard_idx + 1)])