import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.msa = nn.MultiheadAttention(dim, num_heads)
    
    def forward(self, x):
        # x is the patches added with cls token, shape: B, T, num_patches + 1, dim
        B, T, N, D = x.shape
        x = x.permute(0,2,1,3)      # B, num_patches+1, T, dim
        x = x.reshape(B*N, T, D)    # B*(num_patches+1), T, dim

        x, _= self.msa(x,x,x)
        return x.reshape(B,N,T,D).permute(0,2,1,3)
    
class SpatialMultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.msa = nn.MultiheadAttention(dim, num_heads)
    
    def forward(self, x):
        B, T, N, D = x.shape
        x = x.reshape(B * T, N, D).shape
        x, _ = self.msa(x,x,x)
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
        temporal_attention, _ = self.temporal_msa(x,x,x)
        x = self.norm(x + temporal_attention)

        spatial_attention, _ = self.spatial_msa(x,x,x)
        x = self.norm(x + spatial_attention)

        x = self.norm(self.mlp(x) + x)
        return x