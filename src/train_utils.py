import torch.nn as nn
import math

def check_specific_gradients(model):
    stats = {
        'total_trainable_params': 0,
        'blocks_params': 0,
        'temporal_adapter_params': 0,
        'spatial_adapter_params': 0,
        'classifier_params': 0,
        'other_params': 0
    }
    
    with open("logs/gradients.txt", "a+") as f:
        f.write("\nParameters with gradients enabled:\n")
        f.write("=" * 50 + '\n')
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_count = param.numel()
                stats['total_trainable_params'] += param_count
                
                if 'blocks' in name:
                    stats['blocks_params'] += param_count
                elif 'temporal_adapters' in name:
                    stats['temporal_adapter_params'] += param_count
                elif 'spatial_adapters' in name:
                    stats['spatial_adapter_params'] += param_count
                elif 'classifiers' in name:
                    stats['classifier_params'] += param_count
                else:
                    stats['other_params'] += param_count
                f.write(f"\nParameter: {name}\n")
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    f.write(f"Gradient norm: {grad_norm:.6f}\n")
                else:
                    f.write("No gradients computed yet\n")
        f.write("\nParameter Statistics Summary:\n")
        f.write("=" * 50 + '\n')
        f.write(f"Total trainable parameters: {stats['total_trainable_params']:,}\n")
        f.write(f"Block parameters: {stats['blocks_params']:,}\n")
        f.write(f"Temporal adapter parameters: {stats['temporal_adapter_params']:,}\n")
        f.write(f"Spatial adapter parameters: {stats['spatial_adapter_params']:,}\n")
        f.write(f"Classifier parameters: {stats['classifier_params']:,}\n")
        f.write(f"Other parameters: {stats['other_params']:,}\n\n")
    
    return stats

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
