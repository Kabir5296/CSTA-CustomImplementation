import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import Adapter, TimesFormerBlock
from dataclasses import dataclass
from typing import Optional
from einops import rearrange

@dataclass
class CSTAOutput:
    logits: torch.FloatTensor = None
    predictions: torch.FloatTensor = None
    ce_loss: Optional[torch.FloatTensor] = None
    distil_loss: Optional[torch.FloatTensor] = None
    lt_loss: Optional[torch.FloatTensor] = None
    ls_loss: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    accuracy: Optional[torch.FloatTensor] = None

class CSTA(nn.Module):
    def __init__(self,
                 num_classes,
                 num_frames = 8, 
                 img_size = 256, 
                 patch_size = 16, 
                 dim = 768, 
                 num_layers=12, 
                 num_channels = 3,
                 num_heads = 8,
                 init_with_adapters = True,
                 calculate_distil_loss = False,
                 calculate_lt_ls_loss = False,
                 miu_d = 1.0,
                 miu_t = 1.0,
                 miu_s = 1.0,
                 **kwargs,
                 ):
        super().__init__()
        """
        CSTA model class.
        Args:
            num_frames: number of frames in video
            img_size: size of input image, must be square shape
            patch_size: size of patch to create from frames
            dim: dimension of the model
            num_classes: number of classes to initialize first classifier
            num_layers: number of TimesFormer blocks
            num_channels: number of channels in the input image, typically 3
            num_heads: number of heads in attention, typically 8
            init_with_adapters: whether to initialize the model with adapters. if true, default one adapter is added in the blocks.
            calculate_distil_loss: whether to calculate the distillation loss. if true, distil calculation is done on the model without one last adapter. time complexity increases.
            calculate_lt_ls_loss: whether to calculate the lt/ls losses. currently not implemented.
            miu_d: weight of the distillation loss (hyperparameter)
            miu_t: weight of the temporal loss (hyperparameter)
            miu_s: weight of the spatial loss (hyperparameter)

        Methods:
            add_one_adapter_per_block: adds one adapter to each block in the model
            add_one_new_classifier: adds one new classifier to the model
            add_new_task_components: adds one adapter and one classifier to the model, sets calculate_distil_loss to True.
            freeze_all_but_last: freezes all blocks, all adapters except the last, all classifiers except the last
            get_distil_loss: calculates the distillation loss
            forward: forward pass of the model
        """
        self.dim = dim
        self.img_size = img_size
        self.calculate_distil_loss = calculate_distil_loss
        self.calculate_lt_ls_loss = calculate_lt_ls_loss
        self.miu_d = miu_d
        self.miu_t = miu_t
        self.miu_s = miu_s
        self.num_frames = num_frames

        # process video to patches and add positional embeddings, class token
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(num_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.norm = nn.LayerNorm(dim)
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, self.num_frames, 1, self.dim))
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))

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
        self.calculate_lt_ls_loss = True
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

        patch_scale = 0.1
        x = x.reshape(B * T, C, H, W)                       # reshape to (B * T, C, H, W) for patch embedding
        x = self.patch_embed(x)                             # shape: B*T, dim, H//patch_size, W//patch_size
        x = x.flatten(2).transpose(1, 2)                    # shape: B*T, num_patches, dim : num_patches = (H/patch_size)*(W/patch_size)
        # x = x.view(B, T, self.num_patches, self.dim)        # shape: B, T, num_patches, dim

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)           # shape: B*T, 1, dim
        x = torch.cat((cls_tokens, x), dim=1)                           # shape: B*T, num_patches+1, dim
        
        x = x.reshape([B,T,self.num_patches+1,self.dim])                # reshape for adding positional embeddings
        x = x + self.spatial_pos_embed + self.temporal_pos_embed
        x = x.reshape([B*T,self.num_patches+1,self.dim])                # reshape back to: B*T, num_patches+1, dim
        
        loss = ce_loss = distil_loss = lt_loss = ls_loss = accuracy = None
        x_old = x       # B*T, num_patches + 1(cls), dim

        adapter_scale = 0.1
        for block_idx, block in enumerate(self.blocks):
            # x goes to t_msa.
            # t_msa output goes to all adapters (stored in temporal adapter features list)
            # the final temporal output is the sum of all temporal adapter outputs, plus the t_msa and the input normalized
            block_t_msa = block.temporal_msa(x, B, T, self.num_patches)
            temporal_adapter_features = []
            for temporal_adapters in self.temporal_adapters[block_idx]:
                temporal_adapter_features.append(adapter_scale*temporal_adapters(block_t_msa))
            x = self.norm(x + block_t_msa + sum(temporal_adapter_features))

            # same as before but for spatial msa and adapters
            block_s_msa = block.spatial_msa(x)
            spatial_adapter_features = []
            for spatial_adapters in self.spatial_adapters[block_idx]:
                spatial_adapter_features.append(adapter_scale*spatial_adapters(block_s_msa))
            x = self.norm(x + block_s_msa + sum(spatial_adapter_features))

            # final MLP
            res = x
            x = self.norm(x)
            x = res + block.mlp(x)
            x = self.norm(x)

            if self.calculate_distil_loss and targets is not None:
                block_t_msa_old = block.temporal_msa(x_old, B, T, self.num_patches)
                temporal_adapter_features_old = []
                for temporal_adapters in self.temporal_adapters[block_idx][:-1]:
                    temporal_adapter_features_old.append(temporal_adapters(block_t_msa_old))
                x_old = self.norm(x_old + block_t_msa_old + sum(temporal_adapter_features_old))

                block_s_msa_old = block.spatial_msa(x_old)
                spatial_adapter_features_old = []
                for spatial_adapters in self.spatial_adapters[block_idx][:-1]:
                    spatial_adapter_features_old.append(spatial_adapters(block_s_msa_old))
                x_old = self.norm(x_old + block_s_msa_old + sum(spatial_adapter_features_old))

                x_old = self.norm(block.mlp(x_old) + x_old)

        x = x[:, :1, :].squeeze(1).reshape([B,T,self.dim]).mean(dim=1) 
        x_old = x_old[:, :1, :].squeeze(1).reshape([B,T,self.dim]).mean(dim=1)

        outputs = []
        outputs_old = []

        for classifier in self.classifiers:
            outputs.append(classifier(x))
            if self.calculate_distil_loss and targets is not None:
                outputs_old.append(classifier(x_old))

        final_logits = torch.cat(outputs, dim=-1)
        predictions = torch.softmax(final_logits, dim=-1)
        total_loss = []
        if targets is not None:
            accuracy = (predictions.argmax(-1) == targets).float().mean()
            # import pdb
            # pdb.set_trace()
            ce_loss = F.cross_entropy(predictions, targets)
            total_loss.append(ce_loss)
            if self.calculate_distil_loss and targets is not None:
                distil_loss = self.get_distil_loss(torch.cat(outputs_old, dim=1), final_logits)
                total_loss.append(self.miu_d * distil_loss) if distil_loss is not None else None
            loss = sum(total_loss)

        return CSTAOutput(
            logits = final_logits,
            loss = loss,
            ce_loss = ce_loss,
            distil_loss = distil_loss,
            lt_loss = lt_loss,
            ls_loss = ls_loss,
            predictions = predictions,
            last_hidden_state = x,
            accuracy = accuracy,
        )