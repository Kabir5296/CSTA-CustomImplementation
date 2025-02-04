import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import Adapter, AttentionBlock, MLP, AdaptedTimesformerBlock

class CSTA(nn.Module):
    def __init__(self, num_frames, img_size, patch_size, dim, num_classes, num_layers=12, num_channels = 3):
        super().__init__()
        self.dim = dim
        self.img_size = img_size

        # process video to patches and add positional embeddings, class token
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(num_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(self.num_patches * num_frames + 1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        # keeping a list of adapters. Each transformer block has a list of adapters
        self.temporal_adapters = nn.ModuleList([nn.ModuleList() for _ in range(num_layers)])
        self.spatial_adapters = nn.ModuleList([nn.ModuleList() for _ in range(num_layers)])

        # keeping a list of transformer blocks
        self.blocks = nn.ModuleList([AdaptedTimesformerBlock(dim) for _ in range(num_layers)])

        # keeping a list of classifiers
        self.classifiers = nn.ModuleList([nn.Linear(dim, num_classes)])

        # tracking current task for use
        self.current_task = 0

    def add_new_task_components(self, num_new_classes):
        new_classifier = nn.Linear(self.classifiers[-1].in_features, num_new_classes)
        self.classifiers.append(new_classifier)
        
        dim = self.pos_embed.shape[-1]
        for block_idx in range(len(self.blocks)):
            self.temporal_adapters[block_idx].append(Adapter(dim))
            self.spatial_adapters[block_idx].append(Adapter(dim))

        self.current_task += 1
        self.freeze_old_components()

    def freeze_old_components(self):
        for classifier in self.classifiers[:-1]:
            for param in classifier.parameters():
                param.requires_grad = False

        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

        for block_idx in range(len(self.blocks)):
            for task_idx in range(self.current_task - 1):
                for param in self.temporal_adapters[block_idx][task_idx].parameters():
                    param.requires_grad = False

            for task_idx in range(self.current_task - 1):
                for param in self.spatial_adapters[block_idx][task_idx].parameters():
                    param.requires_grad = False

    def forward(self, x):
        B, T, C, H, W = x.shape

        if H != W:
            raise ValueError('Input tensor must have equal height and width')
        elif H != self.img_size:
            raise ValueError('Input tensor has incorrect height and width')

        x = x.reshape(B * T, C, H, W)       # reshape to (B * T, C, H, W) for patch embedding
        x = self.patch_embed(x)             # shape: B*T, dim, H//patch_size, W//patch_size
        x = x.flatten(2).transpose(1, 2)    # shape: B*T, (H//patch_size)*(W//patch_size), dim
        x = x.reshape(B, -1, self.dim)      # shape: B, T*(H//patch_size)*(W//patch_size), dim
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for block_idx, block in enumerate(self.blocks):
            temp_attn_output, _ = block.temporal_msa.msa(x, x, x)
            for task_adapter in self.temporal_adapters[block_idx]:
                temp_attn_output = temp_attn_output + task_adapter(temp_attn_output)
            x = block.temporal_msa.norm(x + temp_attn_output)
            
            spatial_attn_output, _ = block.spatial_msa.msa(x, x, x)
            for task_adapter in self.spatial_adapters[block_idx]:
                spatial_attn_output = spatial_attn_output + task_adapter(spatial_attn_output)
            x = block.spatial_msa.norm(x + spatial_attn_output)
            x = block.norm(x + block.mlp(x))

        x = x[:, 0]
        outputs = []
        for classifier in self.classifiers:
            outputs.append(classifier(x))
            
        final_logits = torch.cat(outputs, dim=1)
        predictions = torch.softmax(final_logits, dim=1)
        
        return predictions