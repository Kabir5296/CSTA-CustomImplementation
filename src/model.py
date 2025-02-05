import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import Adapter, TimesFormerBlock
from dataclasses import dataclass
from typing import Optional

@dataclass
class CSTAOutput:
    logits: torch.FloatTensor = None
    predictions: torch.FloatTensor = None
    ce_loss: Optional[torch.FloatTensor] = None
    distil_loss: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None

class CSTA(nn.Module):
    def __init__(self, 
                 num_frames, 
                 img_size, 
                 patch_size, 
                 dim, 
                 num_classes, 
                 num_layers=12, 
                 num_channels = 3,
                 num_heads = 8,
                 init_with_adapters = True,
                 calculate_distil_loss = False,
                 ):
        super().__init__()
        self.dim = dim
        self.img_size = img_size
        self.calculate_distil_loss = calculate_distil_loss

        # process video to patches and add positional embeddings, class token
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(num_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(self.num_patches * num_frames + 1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.norm = nn.LayerNorm(dim)

        # keeping a list of adapters. Each transformer block has a list of adapters
        self.temporal_adapters = nn.ModuleList([nn.ModuleList() for _ in range(num_layers)])
        self.spatial_adapters = nn.ModuleList([nn.ModuleList() for _ in range(num_layers)])

        # keeping a list of transformer blocks
        self.blocks = nn.ModuleList([TimesFormerBlock(dim = dim, num_heads=num_heads) for _ in range(num_layers)])

        # keeping a list of classifiers
        self.classifiers = nn.ModuleList([nn.Linear(dim, num_classes)])

        # add the first adapter to all the timesformer blocks
        if init_with_adapters:
            self.add_one_adapter_per_block()

        # keep the numbers stored somewhere
        self.model_attributes = self.get_numbers()

    def get_numbers(self):
        total_blocks = len(self.blocks)
        total_temporal_adapters = total_spatial_adapters = 0

        for block_idx in range(len(self.blocks)):
            total_temporal_adapters += len(self.temporal_adapters[block_idx])
            total_spatial_adapters += len(self.spatial_adapters[block_idx])
        adapters_per_block = int(total_spatial_adapters/len(self.blocks))
        total_classifiers = len(self.classifiers)
        current_task = adapters_per_block - 1
        return {
            "total_blocks": total_blocks,
            "adapters_per_block": adapters_per_block,
            "current_task": current_task,
            "total_temporal_adapters": total_temporal_adapters,
            "total_spatial_adapters": total_spatial_adapters,
            "total_classifiers": total_classifiers,
            "total_adapters": total_temporal_adapters + total_spatial_adapters,
            "total_params": sum(p.numel() for p in self.parameters()),
        }

    def add_one_adapter_per_block(self):
        for block_idx in range(len(self.blocks)):
            self.temporal_adapters[block_idx].append(Adapter(self.dim))
            self.spatial_adapters[block_idx].append(Adapter(self.dim))
        self.model_attributes = self.get_numbers()

    def add_one_new_classifier(self, num_new_classes):
        new_classifier = nn.Linear(self.classifiers[-1].in_features, num_new_classes)
        self.classifiers.append(new_classifier)
        self.model_attributes = self.get_numbers()

    def add_new_task_components(self, num_new_classes):
        self.add_one_adapter_per_block()
        self.add_one_new_classifier(num_new_classes)
        self.calculate_distil_loss = True
        self.model_attributes = self.get_numbers()

    def freeze_all_but_last(self):
        for classifier in self.classifiers[:-1]:
            for param in classifier.parameters():
                param.requires_grad = False

        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

        for block_idx in range(len(self.blocks)):
            for adapter in self.temporal_adapters[block_idx][:-1]:
                for param in adapter.parameters():
                    param.requires_grad = False

            for adapter in self.spatial_adapters[block_idx][:-1]:
                for param in adapter.parameters():
                    param.requires_grad = False

    def get_distil_loss(self, old_logits, new_logits):
        return F.kl_div(F.log_softmax(new_logits, dim=1), F.softmax(old_logits, dim=1), reduction='batchmean')

    def forward(self, x, targets=None):
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

        temporal_features = []
        spatial_features = []

        ce_loss = distil_loss = lt_loss = ls_loss = None
        x_old = x
        for block_idx, block in enumerate(self.blocks):
            # x goes to t_msa.
            # t_msa output goes to all adapters (stored in temporal adapter features list)
            # the final temporal output is the sum of all temporal adapter outputs, plus the t_msa and the input normalized
            block_t_msa, _ = block.temporal_msa(x,x,x)
            temporal_adapter_features = []
            for temporal_adapters in self.temporal_adapters[block_idx]:
                temporal_adapter_features.append(temporal_adapters(block_t_msa))
            x = self.norm(x + block_t_msa + sum(temporal_adapter_features))
            temporal_features.append(block_t_msa + sum(temporal_adapter_features))

            # same as before but for spatial msa and adapters
            block_s_msa, _ = block.spatial_msa(x,x,x)
            spatial_adapter_features = []
            for spatial_adapters in self.spatial_adapters[block_idx]:
                spatial_adapter_features.append(spatial_adapters(block_s_msa))
            x = self.norm(x + block_s_msa + sum(spatial_adapter_features))
            spatial_features.append(block_s_msa + sum(spatial_adapter_features))

            # final MLP
            x = self.norm(block.mlp(x) + x)

            if (self.training or self.calculate_distil_loss) and targets is not None:
                block_t_msa_old, _ = block.temporal_msa(x_old,x_old,x_old)
                temporal_adapter_features_old = []
                for temporal_adapters in self.temporal_adapters[block_idx][:-1]:
                    temporal_adapter_features_old.append(temporal_adapters(block_t_msa_old))
                x_old = self.norm(x_old + block_t_msa_old + sum(temporal_adapter_features_old))

                block_s_msa_old, _ = block.spatial_msa(x_old,x_old,x_old)
                spatial_adapter_features_old = []
                for spatial_adapters in self.spatial_adapters[block_idx][:-1]:
                    spatial_adapter_features_old.append(spatial_adapters(block_s_msa_old))
                x_old = self.norm(x_old + block_s_msa_old + sum(spatial_adapter_features_old))

                x_old = self.norm(block.mlp(x_old) + x_old)

        temporal_features = torch.stack(temporal_features, dim=1)
        spatial_features = torch.stack(spatial_features, dim=1)

        x = x[:, 0]
        x_old = x_old[:, 0]
        outputs = []
        outputs_old = []

        for classifier in self.classifiers:
            outputs.append(classifier(x))
            if (self.training or self.calculate_distil_loss) and targets is not None:
                outputs_old.append(classifier(x_old))

        final_logits = torch.cat(outputs, dim=1)
        predictions = torch.softmax(final_logits, dim=1)

        # cross entropy loss
        if targets is not None:
            ce_loss = F.cross_entropy(final_logits, targets)
            if (self.training or self.calculate_distil_loss) and targets is not None:
                distil_loss = self.get_distil_loss(torch.cat(outputs_old, dim=1), final_logits)

        return CSTAOutput(
            logits = final_logits,
            ce_loss = ce_loss,
            distil_loss = distil_loss,
            predictions = predictions,
            last_hidden_state = x,
        )