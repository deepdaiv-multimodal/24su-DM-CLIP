# DM-CLIP
DM-CLIP: Knowledge Distillation Transformer to Mamba for efficient CLIP

![](https://github.com/user-attachments/assets/81fc00c2-3f47-41dc-bfb0-1c1e23b29405)


## Abstract
This study addresses the challenge of deploying Contrastive Learning-based CLIP models, which learn the relationship between images and text, in resource-constrained environments due to their high computational complexity and large model size. To overcome this, we propose an approach that enhances the performance of Mamba-based image encoders by applying Knowledge Distillation from Transformer-based ViT models. Experimental results show that the Mamba-based encoder reduces image encoder latency by 49.58% and overall model latency by 40.82%, with only a 0.12% performance loss. Additionally, it demonstrates 6.6% and 19.4% improvements on the SVHN and EuroSAT datasets, respectively, showcasing strengths in sequential pattern processing and high-resolution spatial information learning. This study validates that the lightweight CLIP encoder can be effectively utilized in mobile and edge device environments and suggests future research directions for developing Mamba-based text encoders and enhancing knowledge distillation techniques.


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
