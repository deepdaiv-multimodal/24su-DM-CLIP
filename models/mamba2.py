# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# try:
#     from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
# except ImportError:
#     RMSNormGated, LayerNorm = None, None

# # !mamba2로 바꿔야함
# from mamba_ssm.modules.mamba2 import Mamba2
# from mamba_ssm.models.mixer_seq_simple import MixerModel
# from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel




# # class MambaLMHeadModel(MambaLMHeadModel):
# #     def __init__(self, config):
# #         super().__init__(config)