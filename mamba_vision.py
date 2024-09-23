from mambavision.models.mamba_vision import MambaVision
import open_clip

# CLIP 모델 로드
model, _, preprocess = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k_augreg')
model.eval()

# 11.55M
mamba_encoder = MambaVision(depths=[3, 3, 10, 5],
                    num_heads=[2, 4, 8, 16],
                    window_size=[8, 8, 14, 7],
                    dim=120,
                    in_dim=64,
                    mlp_ratio=4,
                    resolution=224,
                    drop_path_rate=0.2
                    )

# CLIP 모델의 비전 인코더를 Mamba 비전 인코더로 교체
model.visual = mamba_encoder

print("complete")
model.eval()

# 텍스트 인코더 파라미터 수 계산
image_encoder_params = sum(p.numel() for p in model.visual.parameters())
print("Image Encoder parameters:", f"{image_encoder_params:,}")