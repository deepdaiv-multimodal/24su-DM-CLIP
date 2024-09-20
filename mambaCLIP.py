import open_clip
# from models.mamba2 import CLIPMamba2TextEncoder
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
import torch

# CLIP 모델 로드
model, _, preprocess = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k_augreg')
model.eval()

# Mamba2 텍스트 인코더 파라미터 설정
vocab_size = model.token_embedding.weight.shape[0]

# 파라미터  59.98M
output_dim = model.text_projection.shape[1]

config = MambaConfig(
        d_model=512,
        n_layer=12,
        vocab_size=50277,
        ssm_cfg=dict(layer="Mamba2"),
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=16,
    )

device = "cuda"
dtype = torch.float32

mamba_encoder = MambaLMHeadModel(config, device=device, dtype=dtype)

# CLIP 모델의 텍스트 인코더를 Mamba2 텍스트 인코더로 교체
model.text = mamba_encoder

print("complete")
model.eval()

# 텍스트 인코더 파라미터 수 계산
text_encoder_params = sum(p.numel() for p in model.text.parameters())
print("Text Encoder parameters:", f"{text_encoder_params:,}")

# Mamba2TextEncoder 정보 출력
print("Mamba2TextEncoder 정보:")
print(mamba_encoder)

# 각 Mamba 레이어별 파라미터 수 출력
for i, layer in enumerate(mamba_encoder.backbone.layers):
    num_params = sum(p.numel() for p in layer.parameters())
    print(f"Mamba Layer {i+1} - 파라미터 수: {num_params:,}")

# 전체 텍스트 인코더 파라미터 수 출력
total_params = sum(p.numel() for p in mamba_encoder.parameters())
print(f"전체 MambaTextEncoder 파라미터 수: {total_params:,}")
