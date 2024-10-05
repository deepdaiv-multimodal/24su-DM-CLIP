# MobileMCLIP
MobileMCLIP: Faster and Stronger MoblieCLIP with Mamba

![](https://github.com/user-attachments/assets/81fc00c2-3f47-41dc-bfb0-1c1e23b29405)


## Introduction
We have developed a model using a Mamba-based image encoder that offers faster inference speed and better memory efficiency, outperforming MobileCLIP by being approximately 1.8 times faster and 17% stronger (in terms of acc5). The model was trained on a small 3M dataset using the DataCompDR subset and has shown particularly good performance on high-resolution images and datasets requiring 3D understanding. Although it has been trained only on a small dataset so far, we plan to validate it on large-scale datasets in the future.

![](https://github.com/user-attachments/assets/91c4fcf1-4ae3-4918-8af8-92c2ba0112dc)

![](https://github.com/user-attachments/assets/4d4dc135-e6e0-46f1-9fe5-f29636ffdf25)


## setup

```bash
conda create -n clipenv python=3.10
conda activate clipenv
pip install -r requirements.txt

git clone https://github.com/NVlabs/MambaVision.git
cd MambaVision
pip install -e .
cd ..
```


```bash
bash download_imagenet.sh
```

## run
```bash
bash run_datacompdr12m.sh
```
