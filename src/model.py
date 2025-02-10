import torch, pdb, json
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
    spatial_relations: Optional[torch.FloatTensor] = None
    temporal_relations: Optional[torch.FloatTensor] = None

class CSTA(nn.Module):
    def __init__(self,
                 num_classes,
                 num_frames = 8, 
                 img_size = 224, 
                 patch_size = 16, 
                 dim = 480, 
                 num_layers=12, 
                 num_channels = 3,
                 num_heads = 8,
                 init_with_adapters = True,
                 calculate_distil_loss = False,
                 calculate_lt_ls_loss = False,
                 count_relations = False,
                 miu_d = 0.1,
                 miu_t = 0.1,
                 miu_s = 0.1,
                 lambda_1 = 0.2,
                 K=5,
                 temporal_relations_path = None,
                 spatial_relations_path = None,
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
        self.lambda_1 = lambda_1
        self.num_frames = num_frames
        self.K = K
        self.count_relations = count_relations

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
        
        if temporal_relations_path and spatial_relations_path is not None:
            with open(temporal_relations_path, 'r') as f:
                self.temporal_relations = json.load(f)
            with open(spatial_relations_path, 'r') as f:
                self.spatial_relations = json.load(f)

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
        return F.kl_div(F.log_softmax(new_logits, dim=-1), F.softmax(old_logits, dim=-1), reduction='batchmean')
    
    def run_classifiers(self, x):
        outputs = []            
        for classifier in self.classifiers:
            outputs.append(classifier(x))
        final_tensor = torch.cat(outputs, dim=-1)
        return final_tensor

    def process_features(self, features, B, T, classifier):
        if isinstance(features, list):
            features = torch.mean(torch.stack(features, dim=1), dim=1).squeeze(1).reshape([B,T,self.dim]).mean(dim=1)
        clf_f = classifier(features)
        return clf_f

    def get_relations(self, spatial_features, temporal_features, full_features, B, T, classifier):
        clf_s = self.process_features(spatial_features, B, T, classifier)
        clf_t = self.process_features(temporal_features, B, T, classifier)
        clf_full = self.process_features(full_features, B, T, classifier)
        
        cosine_st = F.cosine_similarity(clf_t, clf_s)
        spatial_ratio = clf_s / clf_full
        temporal_ratio = clf_t / clf_full
        Rs, Rt = torch.cat([spatial_ratio, cosine_st.unsqueeze(-1)], dim=-1), torch.cat([temporal_ratio, cosine_st.unsqueeze(-1)], dim=-1)
        return Rs, Rt
    
    def _get_relations(self, clf_s, clf_t, full_features, B,T,classifier):
        clf_full = self.process_features(full_features, B, T, classifier)
        
        cosine_st = F.cosine_similarity(clf_t, clf_s)
        spatial_ratio = clf_s / clf_full
        temporal_ratio = clf_t / clf_full
        Rs, Rt = torch.cat([spatial_ratio, cosine_st.unsqueeze(-1)], dim=-1), torch.cat([temporal_ratio, cosine_st.unsqueeze(-1)], dim=-1)
        return Rs, Rt

    def forward(self, x, targets=None):
        B, T, C, H, W = x.shape

        if H != W:
            raise ValueError('Input tensor must have equal height and width')
        elif H != self.img_size:
            raise ValueError('Input tensor has incorrect height and width')

        x = x.reshape(B * T, C, H, W)                       # reshape to (B * T, C, H, W) for patch embedding
        x = self.patch_embed(x)                             # shape: B*T, dim, H//patch_size, W//patch_size
        x = x.flatten(2).transpose(1, 2)                    # shape: B*T, num_patches, dim : num_patches = (H/patch_size)*(W/patch_size)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)           # shape: B*T, 1, dim
        x = torch.cat((cls_tokens, x), dim=1)                           # shape: B*T, num_patches+1, dim
        
        x = x.reshape([B,T,self.num_patches+1,self.dim])                # reshape for adding positional embeddings
        x = x + self.spatial_pos_embed + self.temporal_pos_embed
        x = x.reshape([B*T,self.num_patches+1,self.dim])                # reshape back to: B*T, num_patches+1, dim
        
        loss = ce_loss = distil_loss = lt_loss = ls_loss = accuracy = None
        x_old = x       # B*T, num_patches + 1(cls), dim

        self.temporal_features = []
        self.spatial_features = []
        self.temporal_features_old = []
        self.spatial_features_old = []
        for block_idx, block in enumerate(self.blocks):
            # x goes to t_msa.
            # t_msa output goes to all adapters (stored in temporal adapter features list)
            # the final temporal output is the sum of all temporal adapter outputs, plus the t_msa and the input normalized
            block_t_msa = block.temporal_msa(x, B, T, self.num_patches)
            temporal_adapter_features = []
            for temporal_adapters in self.temporal_adapters[block_idx]:
                temporal_adapter_features.append(temporal_adapters(block_t_msa))
            x = self.norm(x + block_t_msa + sum(temporal_adapter_features))
            self.temporal_features.append(x[:, :1, :])
            
            # same as before but for spatial msa and adapters
            block_s_msa = block.spatial_msa(x)
            spatial_adapter_features = []
            for spatial_adapters in self.spatial_adapters[block_idx]:
                spatial_adapter_features.append(spatial_adapters(block_s_msa))
            x = self.norm(x + block_s_msa + sum(spatial_adapter_features))
            self.spatial_features.append(x[:, :1, :])
            
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
                self.temporal_features_old.append(x_old[:, :1, :])

                block_s_msa_old = block.spatial_msa(x_old)
                spatial_adapter_features_old = []
                for spatial_adapters in self.spatial_adapters[block_idx][:-1]:
                    spatial_adapter_features_old.append(spatial_adapters(block_s_msa_old))
                x_old = self.norm(x_old + block_s_msa_old + sum(spatial_adapter_features_old))
                self.spatial_features_old.append(x_old[:, :1, :])

                res = x_old
                x_old = self.norm(x_old)
                x_old = res + block.mlp(x_old)
                x_old = self.norm(x_old)

        x = x[:, :1, :].squeeze(1).reshape([B,T,self.dim]).mean(dim=1)
        self.full_features = x
        final_logits = self.run_classifiers(x)
        predictions = torch.softmax(final_logits, dim=-1)

        spatial_relations, temporal_relations = None, None
        if self.count_relations:
            spatial_relations, temporal_relations = self.get_relations(self.spatial_features, self.temporal_features, self.full_features, B, T, self.classifiers[-1])
        
        total_loss = []
        if targets is not None:
            accuracy = (predictions.argmax(-1) == targets).float().mean()
            ce_loss = F.cross_entropy(final_logits, targets)
            total_loss.append(ce_loss)
            x_old = x_old[:, :1, :].squeeze(1).reshape([B,T,self.dim]).mean(dim=1)
            
            if self.calculate_distil_loss:
                final_logits_old = self.run_classifiers(x_old)
                distil_loss = self.get_distil_loss(final_logits_old, final_logits)
                total_loss.append(self.miu_d * distil_loss) if distil_loss is not None else None
            
            if self.calculate_lt_ls_loss:
                self.temporal_relations = {int(key): torch.tensor(value, device = next(self.parameters()).device) for key, value in self.temporal_relations.items()}
                self.spatial_relations = {int(key): torch.tensor(value, device = next(self.parameters()).device) for key, value in self.spatial_relations.items()}
                self.full_features_old = x_old
                
                if len(self.classifiers) < 2:
                    raise ValueError(f"Classifiers have {len(self.classifiers)} modules, meaning its not ready for current task.")
                
                # To calculat Lt and Ls losses, first, we need Rsn and Rtn as well as Rsn_old, Rtn_old
                Rsn, Rtn = self.get_relations(self.spatial_features, self.temporal_features, self.full_features, B, T, self.classifiers[-1])
                Rsn_1, Rtn_1 = self.get_relations(self.spatial_features_old, self.temporal_features_old, self.full_features_old, B, T, self.classifiers[-2])

                # R are of shape B, something [usually classes + 1]
                # count the top k similarities
                temporal_similarities = []
                for _, value in self.temporal_relations.items():
                    temporal_similarities.append(F.cosine_similarity(Rtn_1, value.unsqueeze(0)))
                top_k_temporal_sum = torch.sum(torch.topk(torch.stack(temporal_similarities).reshape([B, len(self.temporal_relations)]), k=self.K, dim=1)[0], dim=1)
                
                spatial_similarities = []
                for _, value in self.spatial_relations.items():
                    spatial_similarities.append(F.cosine_similarity(Rsn_1, value.unsqueeze(0)))
                top_k_spatial_sum = torch.sum(torch.topk(torch.stack(spatial_similarities).reshape([B, len(self.spatial_relations)]), k=self.K, dim=1)[0], dim=1)
                
                SnK = self.process_features(self.spatial_features, B, T, self.classifiers[-1]) * top_k_spatial_sum.reshape([B,1])
                TnK = self.process_features(self.temporal_features, B, T, self.classifiers[-1]) * top_k_temporal_sum.reshape([B,1])
                
                # SnK and TnK have dimensions B, classes since they're nothing but classifiers output mul by a scalar
                RsnK, RtnK = self._get_relations(SnK, TnK, self.full_features, B, T, self.classifiers[-1])
                
                lt_loss = (1 - F.cosine_similarity(RtnK, Rtn)).mean()
                ls_loss = (1 - F.cosine_similarity(RsnK, Rsn)).mean()
                total_loss.append(self.miu_t * lt_loss) if lt_loss is not None else None
                total_loss.append(self.miu_s * ls_loss) if ls_loss is not None else None
            
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
            spatial_relations=spatial_relations,
            temporal_relations=temporal_relations
        )
