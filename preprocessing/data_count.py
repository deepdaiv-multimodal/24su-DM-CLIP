import tarfile
import os

def extract_uids_from_tar(tar_file_path):
    """Extract UIDs from .jpg files inside the tar file."""
    uids = set()
    
    # tar 파일 열기
    with tarfile.open(tar_file_path, 'r') as tar:
        for member in tar.getmembers():
            # .jpg 파일만 처리
            if member.isfile() and member.name.endswith(".jpg"):
                # 파일 이름의 첫 번째 부분이 UID라고 가정
                uid = os.path.basename(member.name).split('.')[0]
                uids.add(uid)
    
    return uids

def count_unique_uids(data_path, num_files=501):
    """Count unique UIDs across multiple tar files."""
    all_uids = set()

    # 00000000 ~ 00000499까지 tar 파일 처리
    for i in range(num_files):
        tar_file_path = data_path.format(i)
        if os.path.exists(tar_file_path):
            uids_in_tar = extract_uids_from_tar(tar_file_path)
            all_uids.update(uids_in_tar)
            print(f"Processed {tar_file_path}: found {len(uids_in_tar)} UIDs")
        else:
            print(f"File not found: {tar_file_path}")

    print(f"Total unique UIDs: {len(all_uids)}")
    return all_uids

# 경로 지정
data_path = "DataCompDR/{:08d}.tar"
unique_uids = count_unique_uids(data_path)
