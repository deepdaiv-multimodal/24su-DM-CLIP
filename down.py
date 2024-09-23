import os
import requests
from tqdm import tqdm

# 데이터셋 파일의 기본 URL 설정
base_url = "https://huggingface.co/datasets/apple/DataCompDR-12M/resolve/main/"
# base_url = "https://huggingface.co/datasets/mlfoundations/DataComp-12M/resolve/main/"

# 다운로드 디렉토리 지정
download_dir = "DataCompDR-6M"

# 다운로드 디렉토리 생성 (존재하지 않을 경우)
os.makedirs(download_dir, exist_ok=True)

# 파일 인덱스 범위 설정 (00000000.tar부터 00000500.tar까지)
for i in range(0, 501):  # 0부터 500까지 포함
    file_index = f"{i:08d}"
    filename = f"{file_index}.tar"
    url = base_url + filename
    print(f"{filename} 다운로드 중...")

    # 파일을 저장할 전체 경로 설정
    file_path = os.path.join(download_dir, filename)

    # 이미 파일이 존재하는지 확인하여 중복 다운로드 방지
    if os.path.exists(file_path):
        print(f"{filename}은(는) 이미 존재합니다. 다운로드를 건너뜁니다.")
        continue

    # 파일 다운로드
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
            for data in response.iter_content(chunk_size=1024):
                progress_bar.update(len(data))
                f.write(data)
            progress_bar.close()
        print(f"{filename} 다운로드 완료.")
    else:
        print(f"{filename} 다운로드 실패. HTTP 상태 코드: {response.status_code}")
