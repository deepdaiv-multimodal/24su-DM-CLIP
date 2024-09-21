# M4-CLIP
M4-CLIP: Mobile Multi Modal Mamba with CLIP

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
