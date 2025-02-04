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

class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.msa = nn.MultiheadAttention(dim, num_heads)
        self.adapter = Adapter(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        attn_output, _ = self.msa(x, x, x)
        adapted = self.adapter(attn_output)
        return self.norm(x + adapted + attn_output)

class MLP(nn.Module):
    def __init__(self, dim, factor=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * factor),
            nn.GELU(),
            nn.Linear(dim * factor, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.mlp(x))

class AdaptedTimesformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, factor=4):
        super().__init__()
        self.temporal_msa = AttentionBlock(dim, num_heads)
        self.spatial_msa = AttentionBlock(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.temporal_msa(x)
        x = self.spatial_msa(x)
        x = self.norm(self.mlp(x) + x)
        return x