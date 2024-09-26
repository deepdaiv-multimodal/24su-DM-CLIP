import os
import tarfile
import requests
from io import BytesIO
from PIL import Image
import shutil
import concurrent.futures
from tqdm import tqdm

def check_pair(uid, temp_img_dir, temp_data6m_dir):
    """Check if both .jpg and .txt files for a UID are present."""
    img_file = os.path.join(temp_img_dir, f"{uid}.jpg")
    txt_file = os.path.join(temp_data6m_dir, f"{uid}.txt")
    
    # Check if both the .jpg and .txt files exist
    return os.path.exists(img_file) and os.path.exists(txt_file)

def process_tar_files(start_idx, end_idx, dr6m_dir, data6m_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(start_idx, end_idx + 1):
        tar_idx = f"{idx:08d}"
        dr6m_tar_path = os.path.join(dr6m_dir, f"{tar_idx}.tar")
        data6m_tar_path = os.path.join(data6m_dir, f"{tar_idx}.tar")
        output_tar_path = os.path.join(output_dir, f"{tar_idx}.tar")
        
        if os.path.exists(output_tar_path):
            print(f"Skipping shard {tar_idx} as it already exists.")
            continue

        # Separate temporary directories for extraction
        temp_img_dir = os.path.join(output_dir, f"temp_img_{tar_idx}")
        temp_dr6m_dir = os.path.join(output_dir, f"temp_dr6m_{tar_idx}")
        temp_data6m_dir = os.path.join(output_dir, f"temp_data6m_{tar_idx}")

        print(f"Processing shard {tar_idx}...")

        # Create temporary directories
        os.makedirs(temp_img_dir, exist_ok=True)
        os.makedirs(temp_dr6m_dir, exist_ok=True)
        os.makedirs(temp_data6m_dir, exist_ok=True)

        try:
            # DataCompDR-6M tar file processing (extract files except .jpg)
            with tarfile.open(dr6m_tar_path, 'r') as dr6m_tar:
                uid_url_map = {}
                for member in dr6m_tar.getmembers():
                    if member.name.endswith('.url.txt'):
                        uid = member.name.split('.')[0]
                        url_file = dr6m_tar.extractfile(member)
                        url = url_file.read().decode('utf-8').strip()
                        uid_url_map[uid] = url

                # Download images to temp_img_dir
                def download_image(uid_url):
                    uid, url = uid_url
                    try:
                        response = requests.get(url, timeout=5)
                        response.raise_for_status()
                        image = Image.open(BytesIO(response.content)).convert('RGB')
                        image_path = os.path.join(temp_img_dir, f"{uid}.jpg")
                        image.save(image_path, 'JPEG')
                        return uid, True
                    except Exception as e:
                        # print(f"Failed to download image for UID {uid}: {e}")
                        return uid, False

                # Parallel image download
                with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                    uid_list = list(uid_url_map.items())
                    results = list(tqdm(executor.map(download_image, uid_list), total=len(uid_list)))

                # 성공적으로 이미지를 다운로드한 UID만 포함
                successful_uids = set(uid for uid, success in results if success)

                # Extract files for successful UIDs from dr6m_tar
                for member in dr6m_tar.getmembers():
                    uid = member.name.split('.')[0]
                    if uid in successful_uids and not member.name.endswith('.jpg'):
                        dr6m_tar.extract(member, temp_dr6m_dir)

            # DataComp-6M tar file processing (extract all related to successful UIDs)
            with tarfile.open(data6m_tar_path, 'r') as data6m_tar:
                for member in data6m_tar.getmembers():
                    uid = member.name.split('.')[0]
                    if uid in successful_uids:
                        data6m_tar.extract(member, temp_data6m_dir)

            # Check for matching .jpg and .txt pairs and log number of pairs
            valid_uids = set()
            for uid in successful_uids:
                if check_pair(uid, temp_img_dir, temp_data6m_dir):
                    valid_uids.add(uid)
                else:
                    print(f"Removing UID {uid} due to missing .jpg or .txt file.")

            # Print the number of valid pairs
            print(f"Total valid pairs for shard {tar_idx}: {len(valid_uids)}")

            # Combine all valid pairs into a new tar file
            with tarfile.open(output_tar_path, 'w') as out_tar:
                for uid in valid_uids:
                    # Add image file
                    img_file_path = os.path.join(temp_img_dir, f"{uid}.jpg")
                    out_tar.add(img_file_path, arcname=f"{uid}.jpg")
                    
                    # Add DR-6M files
                    for ext in ['url.txt', 'json', 'npz', 'paug.json', 'syn.json']:
                        dr6m_file_path = os.path.join(temp_dr6m_dir, f"{uid}.{ext}")
                        out_tar.add(dr6m_file_path, arcname=f"{uid}.{ext}")
                    
                    # Add Data6M .txt file
                    data6m_file_path = os.path.join(temp_data6m_dir, f"{uid}.txt")
                    out_tar.add(data6m_file_path, arcname=f"{uid}.txt")

            print(f"Shard {tar_idx} processed and saved to {output_tar_path}")

            os.remove(dr6m_tar_path)
            os.remove(data6m_tar_path)


        except Exception as e:
            print(f"Error processing shard {tar_idx}: {e}")

        finally:
            # Clean up temporary directories after processing
            shutil.rmtree(temp_img_dir)
            shutil.rmtree(temp_dr6m_dir)
            shutil.rmtree(temp_data6m_dir)


if __name__ == "__main__":
    # Configuration
    start_index = 300
    end_index = 300

    # Set directories (updated paths based on the context you provided)
    dr6m_directory = "DataCompDR-6M"  # DataCompDR-6M tar files
    data6m_directory = "DataComp-6M"  # DataComp-6M tar files
    output_directory = "DataCompDR"   # Output tar files

    process_tar_files(start_index, end_index, dr6m_directory, data6m_directory, output_directory)