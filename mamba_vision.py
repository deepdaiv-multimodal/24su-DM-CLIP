from mambavision.models.mamba_vision import MambaVision
import open_clip
from transformers import AutoModel
import torch

# CLIP 모델 로드
model, _, preprocess = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k_augreg')
model.eval()

# 11.55M
mamba_encoder = AutoModel.from_pretrained("nvidia/MambaVision-T-1K", trust_remote_code=True)

# CLIP 모델의 비전 인코더를 Mamba 비전 인코더로 교체
model.visual = mamba_encoder
if hasattr(model.visual, 'head'):
    del model.visual.head

# 512로 출력하는 새로운 헤드 추가
model.visual.head = torch.nn.Linear(640, 512)
# 1. 비전 인코더의 모든 파라미터 고정 (freeze)
for param in model.visual.parameters():
    param.requires_grad = False

# 2. 새로운 분류기(헤드)의 파라미터만 학습 가능하게 설정
for param in model.visual.head.parameters():
    param.requires_grad = True

print(mamba_encoder)

print("complete")
model.eval()

# 텍스트 인코더 파라미터 수 계산
image_encoder_params = sum(p.numel() for p in model.visual.parameters())
print("Image Encoder parameters:", f"{image_encoder_params:,}")