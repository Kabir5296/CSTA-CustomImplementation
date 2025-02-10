from src.model import CSTA
from train_task0 import VideoDataset
from accelerate import Accelerator
import torch, os, random, accelerate, logging, datetime
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
from abc import abstractmethod
from torchvision.io import read_video
import torch.optim as optim
from warnings import filterwarnings
import torchvision.transforms as transforms
import math

ucf_dataset_test_path = "DATA/UCF101/tasks/task_1/val.csv"
num_of_old_classes = 51
state_dict_path = "Outputs/Models/Task1/Trial2/best_model.pth"

ucf_test = pd.read_csv(ucf_dataset_test_path)

all_labels = sorted(ucf_test['label'].unique().tolist())
id2label = {}
label2id = {}
for index, label in enumerate(all_labels):
    id2label[index + num_of_old_classes] = label
    label2id[label] = index + num_of_old_classes

class CSTAConfig:
    """
    Load the initial model with the EXACT SAME CONFIGS as the previous model.
    In this case the initial model had 50 num_classes
    """
    num_frames = 8                      # taking a lower frame numbers for initial training
    img_size = 224                      # the frames are sized at 256*256
    patch_size = 16                     # patch size
    dim = 480                           # model dimension
    num_classes = 51                    # lets say we have a data for initial training with these classes
    num_layers= 12                      # total number of timesformer layers or blocks
    num_channels = 3                    # RGB
    num_heads = 8                       # using 8 heads in attention
    init_with_adapters = True           # for task 0, the model is initialized with one adapter per block
    calculate_distil_loss = False       # For task 0 training, no distillation loss is needed
    calculate_lt_lss_loss = False       # For task 0 training, no lt ls loss is needed
    miu_d = 0.15                        # distillation loss weight
    miu_t = 0.15                        # lt loss weight (currently not implemented)
    miu_s = 0.15                        # ls loss weight (currently not implemented)
    lambda_1 = 1                        # new classifiers multiplying factor
    K = 5
    temporal_relations_path = "DATA/UCF101/tasks/task_1/temporal_relations.json"
    spatial_relations_path = "DATA/UCF101/tasks/task_1/spatial_relations.json"

class DatasetConfig:
    img_size = CSTAConfig.img_size
    num_frames = CSTAConfig.num_frames
    root_path = "DATA/UCF101"
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    id2label = id2label
    label2id = label2id

class EvalConfigs:
    evaluation_batch_size = 12
    dataloader_pin_memory = False
    dataloader_persistent_workers = True
    dataloader_num_workers = 4

model = CSTA(**vars(CSTAConfig))
model.add_new_task_components(10)
model.freeze_all_but_last()
model.calculate_distil_loss =  model.calculate_lt_ls_loss = False
model.load_state_dict(torch.load(state_dict_path))

test_dataset = VideoDataset(ucf_test,
                            img_size = DatasetConfig.img_size, 
                            mean = DatasetConfig.mean, 
                            std = DatasetConfig.std,
                            num_frames = DatasetConfig.num_frames,
                            root_path = DatasetConfig.root_path,
                            id2label = DatasetConfig.id2label,
                            label2id = DatasetConfig.label2id,
                            )
test_dataloader = DataLoader(test_dataset, 
                            batch_size=EvalConfigs.evaluation_batch_size, 
                            shuffle=False, 
                            pin_memory=EvalConfigs.dataloader_pin_memory, 
                            persistent_workers=EvalConfigs.dataloader_persistent_workers,
                            num_workers=EvalConfigs.dataloader_num_workers,
                            )

def evaluate(model, eval_dataloader, accelerator):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0
    
    progress_bar = tqdm(
        total=len(eval_dataloader),
        disable=not accelerator.is_local_main_process,
        desc=f"Test"
    )
    
    with torch.no_grad():
        for batch in eval_dataloader:
            input_frames = batch["input_frames"]
            labels = batch["label"]
            batch_size = labels.size(0)

            outputs = model(input_frames, labels)
            loss = outputs.loss
            
            predictions = outputs.predictions.argmax(-1)
            correct = (predictions == labels).sum().item()
            accuracy = correct / batch_size
            
            running_loss += loss.item() * batch_size
            running_acc += correct
            total_samples += batch_size
            
            progress_bar.update(1)
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "batch_acc": f"{accuracy:.4f}",
                "running_acc": f"{running_acc/total_samples:.4f}"
            })
    
    avg_loss = running_loss / total_samples
    avg_acc = running_acc / total_samples
    progress_bar.close()
    with open("logs/test.log", "+a") as test_log:
        test_log.write(f"Model Used: {state_dict_path}, Test data path: {ucf_dataset_test_path}\n")
        test_log.write(f"Average Loss: {avg_loss:.4f}, "+f" Average Accuracy: {avg_acc:.4f}, "+f" Samples: {total_samples}\n\n")
    
    return avg_loss, avg_acc

accelerator = Accelerator()
model, test_dataloader = accelerator.prepare(model, test_dataloader)

from warnings import filterwarnings
filterwarnings("ignore")
evaluate(model, test_dataloader, accelerator)