import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
except ImportError:
    RMSNormGated, LayerNorm = None, None

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

class Mamba2TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        d_model=768,
        n_layer=12,
        d_state=16,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        bias=False,
        conv_bias=True,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # Mamba layers
        # 맘바 모듈 적용하기!
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=expand,
                dt_rank=dt_rank,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init=dt_init,
                dt_scale=dt_scale,
                dt_init_floor=dt_init_floor,
                bias=bias,
                conv_bias=conv_bias,
                **kwargs
            ) for _ in range(n_layer)
        ])
        
        # Final layer normalization
        self.norm_f = RMSNormGated(d_model)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.embedding(input_ids) + self.pos_embedding[:, :T, :]
        
        for layer in self.layers:
            x = x + layer(x)  
        
        x = self.norm_f(x)
        return x

class CLIPMamba2TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        output_dim,
        **kwargs
    ):
        super().__init__()
        self.mamba_encoder = Mamba2TextEncoder(vocab_size, max_seq_len, **kwargs)
        self.projection = nn.Linear(self.mamba_encoder.d_model, output_dim)
    
    def forward(self, input_ids):
        x = self.mamba_encoder(input_ids)
        return self.projection(x[:, -1, :]) 