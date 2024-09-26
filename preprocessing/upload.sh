#!/bin/bash

# 데이터셋 이름과 경로 설정
DATASET_NAME="taewan2002/DataCompDR-5M"
LOCAL_PATH="DataCompDR-5M"
REPO_TYPE="dataset"

# 0부터 500까지 파일 업로드
for i in $(seq -f "%08g" 201 300)
do
    FILE_PATH="${LOCAL_PATH}/${i}.tar"
    echo "Uploading $FILE_PATH ..."
    
    huggingface-cli upload $DATASET_NAME $FILE_PATH --private --repo-type $REPO_TYPE

    if [ $? -eq 0 ]; then
        echo "Successfully uploaded $FILE_PATH"
    else
        echo "Failed to upload $FILE_PATH"
    fi
done

echo "All uploads completed."
