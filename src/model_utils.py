import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Adapter(nn.Module):
    def __init__(self, dim, reduction_factor=4):
        super().__init__()
        self.fc_down = nn.Linear(dim, dim // reduction_factor)
        self.fc_up = nn.Linear(dim // reduction_factor, dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc_up(self.gelu(self.fc_down(x)))

class TemporalMultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.layer_norm = nn.LayerNorm(dim)
        self.msa = nn.MultiheadAttention(dim, num_heads)
        self.proj = nn.Linear(dim,dim)
    
    def temporal_preprocess(self, x, B, T, num_patches):
        # take input x in B*T, num_patches+1, dim format. convert to B*num_patches, T, dim and returns
        cls_tokens, x_temporal = x[:, :1, :], x[:, 1:, :]
        x_temporal = x_temporal.reshape([B*num_patches, T, self.dim])        # shape: B*num_patches, T, dim
        return x_temporal, cls_tokens
    
    def forward(self, x, B, T, num_patches):
        # x is the patches added with cls token, shape: B*T, num_patches+1, dim
        x, cls_token = self.temporal_preprocess(x,B,T,num_patches)
        
        res = x
        x = self.layer_norm(x)      # shape: B*num_patches, T, dim
        x, _= self.msa(x,x,x)       # shape: B*num_patches, T, dim
        x = self.proj(x)            # shape: B*num_patches, T, dim
        x = x + res
        
        x = x.reshape([B*T, num_patches, self.dim])     # shape: B*T, num_patches, dim
        x = torch.cat((cls_token, x), dim=1)            # shape: B*T, num_patches+1, dim 
        return x
    
class SpatialMultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.msa = nn.MultiheadAttention(dim, num_heads)
        self.proj = nn.Linear(dim,dim)
    
    def forward(self, x):
        res = x
        x = self.layer_norm(x)
        x, _= self.msa(x,x,x)
        x = self.proj(x)
        x = x + res
        return x

class TimesFormerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, factor=4):
        super().__init__()
        """
        x_temporal = T_MSA(x)
        x = NORM(x_temporal + x)

        x_spatial = S_MSA(x)
        x = NORM(x + x_spatial)

        x = NORM(MLP(x) + x)
        """

        self.temporal_msa = TemporalMultiheadAttention(dim, num_heads)
        self.spatial_msa = SpatialMultiheadAttention(dim, num_heads)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * factor),
            nn.GELU(),
            nn.Linear(dim * factor, dim)
        )
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        temporal_attention = self.temporal_msa(x,x,x)
        x = self.norm(x + temporal_attention)

        spatial_attention = self.spatial_msa(x,x,x)
        x = self.norm(x + spatial_attention)

        x = self.norm(self.mlp(x) + x)
        return x