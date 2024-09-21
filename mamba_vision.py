from mambavision.models.mamba_vision import MambaVision
import open_clip

# CLIP 모델 로드
model, _, preprocess = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k_augreg')
model.eval()

# 11.55M
mamba_encoder = MambaVision(depths=[1, 3, 12, 6],
                    num_heads=[1, 2, 10, 8],
                    window_size=[4, 4, 12, 8],
                    dim=40,
                    in_dim=32,
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