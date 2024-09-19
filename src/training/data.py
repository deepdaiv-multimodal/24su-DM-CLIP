import webdataset as wds
from huggingface_hub import HfFileSystem, get_token, hf_hub_url
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import io
from PIL import Image, UnidentifiedImageError
import requests
import json
from tqdm import tqdm

def preprocess_image(img_bytes, preprocess_img):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        return preprocess_img(img)
    except UnidentifiedImageError:
        print("Warning: Unable to open image. Returning a blank tensor.")
        return torch.zeros((3, 224, 224))

def process_sample(sample, preprocess_img):
    url = sample.get('url.txt', '').strip()
    print(f"Image URL: {url}")

    try:
        response = requests.get(url)
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        image = preprocess_img(image)
    except (requests.exceptions.RequestException, UnidentifiedImageError) as e:
        print(f"Error loading image from URL {url}: {e}")
        image = None

    json_data = sample.get('json', {})
    uid = json_data.get('uid', 'unknown')
    sha256 = json_data.get('sha256', 'unknown')
    print(f"UID: {uid}, SHA256: {sha256}")

    return image

def get_datacomp_12m_dataset(args, preprocess_img, is_train, tokenizer):
    fs = HfFileSystem()
    files = [fs.resolve_path(path) for path in fs.glob("hf://datasets/apple/DataCompDR-12M/**/*.tar")]
    urls = [hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset") for file in files]
    urls = f"pipe: curl -s -L -H 'Authorization:Bearer {get_token()}' {'::'.join(urls)}"

    dataset = (
        wds.WebDataset(urls)
        .decode()
        .map(lambda sample: process_sample(sample, preprocess_img))
    )

    # TQDM을 통해 데이터 로더에 대한 진행 상황 표시
    dataset = list(dataset)
    
    # WebDataset을 DataLoader로 래핑
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )

    steps_per_epoch = getattr(args, 'steps_per_epoch', 1000)
    estimated_size = args.batch_size * steps_per_epoch
    return dataloader, estimated_size

def get_data(args, preprocess_fns, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}
    if args.train_data:
        train_dataset, train_size = get_datacomp_12m_dataset(args, preprocess_train, is_train=True, tokenizer=tokenizer)
        # TQDM을 통해 데이터 로더에 대한 진행 상황 표시
        data["train"] = tqdm(train_dataset, desc='Training Data Loader', total=len(train_dataset))
        data["train_size"] = train_size
    if args.val_data:
        val_dataset, val_size = get_datacomp_12m_dataset(args, preprocess_val, is_train=False, tokenizer=tokenizer)
        # TQDM을 통해 데이터 로더에 대한 진행 상황 표시
        data["val"] = tqdm(val_dataset, desc='Validation Data Loader', total=len(val_dataset))
        data["val_size"] = val_size
    return data


# import webdataset as wds
# from huggingface_hub import HfFileSystem, get_token, hf_hub_url
# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import io
# from PIL import Image, UnidentifiedImageError
# import requests
# def get_datacomp_12m_dataset(args, preprocess_img, is_train, tokenizer):
#     # HuggingFace 파일시스템 설정
#     fs = HfFileSystem()

#     # DataComp-12M 데이터셋의 tar 파일 목록 가져오기
#     files = [fs.resolve_path(path) for path in fs.glob("hf://datasets/apple/DataCompDR-12M/*/.tar")]

#     # URL 생성
#     urls = [hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset") for file in files]
#     urls = f"pipe: curl -s -L -H 'Authorization:Bearer {get_token()}' {'::'.join(urls)}"
#     dataset = wds.WebDataset(urls).decode()
#     # DataLoader 설정
#     dataloader = DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         num_workers=args.workers,
#         pin_memory=True,
#     )
#     steps_per_epoch = getattr(args, 'steps_per_epoch', 1000)  # args에 없으면 기본값 1000 사용
#     estimated_size = args.batch_size * steps_per_epoch
#     return dataloader, estimated_size
# def get_data(args, preprocess_fns, tokenizer=None):
#     preprocess_train, preprocess_val = preprocess_fns
#     data = {}
#     if args.train_data:
#         data["train"], data["train_size"] = get_datacomp_12m_dataset(args, preprocess_train, is_train=True, tokenizer=tokenizer)
#     if args.val_data:
#         data["val"], data["val_size"] = get_datacomp_12m_dataset(args, preprocess_val, is_train=False, tokenizer=tokenizer)
#     return data