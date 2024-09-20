import os
import requests
from io import BytesIO
from PIL import Image
import webdataset as wds
from tqdm import tqdm
from huggingface_hub import HfFileSystem, get_token, hf_hub_url
import json

def preprocess_sample(sample, tokenizer):
    """
    WebDataset 샘플을 전처리합니다.
    스트리밍 방식으로 이미지를 다운로드하고 텍스트를 토크나이징합니다.

    Args:
        sample: WebDataset 샘플 (이미지 URL, 텍스트 포함)
        tokenizer: 텍스트 토크나이저

    Returns:
        전처리된 이미지 및 토큰화된 텍스트
    """
    try:
        # sample은 (image_url, text) 형식의 튜플일 것으로 가정
        image_url = sample[0]  # '__url__' 대신 0번 인덱스
        text_file = sample[1]  # 'syn.json' 대신 1번 인덱스


        # 이미지 다운로드
        response = requests.get(image_url, headers={"Authorization": f"Bearer {get_token()}"}, timeout=5)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")

        # 텍스트 전처리 (JSON 파일)
        text = json.loads(text_file)
        
        return image, tokenizer(text)
    
    
    except Exception as e:
        print(f"Error processing sample: {e}")
        return None

def print_sample_keys(dataset):
    for sample in dataset:
        print(sample.keys())
        break  # 첫 번째 샘플만 출력하고 종료


def get_webdataset(args, preprocess_fn, tokenizer):
    """
    WebDataset을 스트리밍 방식으로 가져옵니다.

    Args:
        args: 데이터셋 및 배치 관련 인자
        preprocess_fn: 샘플을 전처리하는 함수
        tokenizer: 텍스트 토크나이저

    Returns:
        WebDataset dataloader
    """
    # Hugging Face dataset에서 WebDataset tar 파일 목록 가져오기
    fs = HfFileSystem()
    files = [fs.resolve_path(path) for path in fs.glob("hf://datasets/apple/DataCompDR-12M/**/*.tar")]
    urls = [hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset") for file in files]
    
    # 데이터셋을 가져와서 키를 확인합니다.
    dataset = wds.WebDataset(urls)
    # print_sample_keys(dataset)

    # WebDataset을 스트리밍 방식으로 불러오기
    dataset = (
        wds.WebDataset(urls)
        .decode("pil", handler=wds.handlers.warn_and_continue)
        .to_tuple("__url__", "syn.json")  # 필드 이름이 아닌 인덱스를 사용할 수 있음
        .map(lambda sample: preprocess_fn(sample, tokenizer))  # 전처리 함수 적용
        .batched(args.batch_size)  # 배치로 묶기
    )

    return dataset

def get_data(args, preprocess_fns, tokenizer=None):
    """
    데이터셋을 WebDataset 스트리밍 방식으로 불러옵니다.

    Args:
        args: 데이터셋 및 배치 관련 인자
        preprocess_fns: 전처리 함수들 (train, validation용)
        tokenizer: 텍스트 토크나이저

    Returns:
        train, validation 데이터 로더
    """
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_webdataset(args, preprocess_train, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_webdataset(args, preprocess_val, tokenizer=tokenizer)

    return data

# 실행 예시
if __name__ == "__main__":
    class Args:
        start_shard_idx = 0
        end_shard_idx = 1
        batch_size = 32
        train_data = True
        val_data = False

    args = Args()
    tokenizer = lambda x: x  # 실제로는 CLIPTokenizer 같은 텍스트 토크나이저를 사용

    # 전처리 함수 (이미지 전처리, 텍스트 토크나이징 등)
    preprocess_train_fn = preprocess_sample
    preprocess_val_fn = preprocess_sample

    data_loaders = get_data(args, (preprocess_train_fn, preprocess_val_fn), tokenizer=tokenizer)

    # 예시: train 데이터셋에서 배치 처리
    for batch in tqdm(data_loaders["train"], desc="Training data processing"):
        images, texts = batch  # 배치에서 이미지와 텍스트 가져오기
        # 여기서 이미지를 모델에 넣어 학습 가능
        print(f"Batch images: {len(images)}, Batch texts: {len(texts)}")
