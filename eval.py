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

logger = logging.getLogger(__name__)
ucf_dataset_test_path = "DATA/UCF101/tasks/task_0/test.csv"

# ucf_dataset_training_path = "DATA/UCF101/tasks/task_beta_0/train.csv"
# ucf_dataset_test_path = "DATA/UCF101/tasks/task_beta_0/test.csv"
# ucf_dataset_valid_path = "DATA/UCF101/tasks/task_beta_0/val.csv"

ucf_test = pd.read_csv(ucf_dataset_test_path)

all_labels = sorted(ucf_test['label'].unique().tolist())
id2label = {}
label2id = {}
for index, label in enumerate(all_labels):
    id2label[index] = label
    label2id[label] = index

state_dict_path = "Outputs/Models/Trial_51_run2/best_model.pth"

class CSTAConfig:
    num_frames = 8                 # taking a lower frame numbers for initial training
    img_size = 224                 # the frames are sized at 256*256
    patch_size = 16                # patch size
    dim = 480                      # model dimension
    num_classes = len(all_labels)  # lets say we have a data for initial training with these classes
    num_layers= 12                 # total number of timesformer layers or blocks
    num_channels = 3               # RGB
    num_heads = 8                  # using 8 heads in attention
    init_with_adapters = True      # for task 0, the model is initialized with one adapter per block
    calculate_distil_loss = False  # For task 0 training, no distillation loss is needed
    calculate_lt_lss_loss = False  # For task 0 training, no lt ls loss is needed
    miu_d = 0.1                    # distillation loss weight
    miu_t = 0.1                    # lt loss weight (currently not implemented)
    miu_s = 0.1                    # ls loss weight (currently not implemented)
    lambda_1 = 0.2

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

