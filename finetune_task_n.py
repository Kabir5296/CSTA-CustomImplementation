from src.model import CSTA
from accelerate import Accelerator
import torch, os, random, accelerate, logging, datetime
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from abc import abstractmethod
from torchvision.io import read_video
import torch.optim as optim
from warnings import filterwarnings
import torchvision.transforms as transforms

from train_task0 import VideoDataset, train_epoch, evaluate, set_all_seeds

logging.basicConfig(
    filename="logs/train.log",
    filemode="a",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

ucf_dataset_training_path_task0 = "DATA/UCF101/tasks/task_0/train.csv"
ucf_dataset_test_path_task0 = "DATA/UCF101/tasks/task_0/test.csv"
ucf_dataset_valid_path_task0 = "DATA/UCF101/tasks/task_0/val.csv"

ucf_task1_train = "DATA/UCF101/tasks/task_1/train.csv"
ucf_task1_valid = "DATA/UCF101/tasks/task_1/val.csv"
ucf_task1_test = "DATA/UCF101/tasks/task_1/test.csv"

train_ucf_task1 = pd.read_csv(ucf_task1_train)
valid_ucf_task1 = pd.read_csv(ucf_task1_valid)
ucf_train_task0 = pd.read_csv(ucf_dataset_training_path_task0)
ucf_valid_task0 = pd.read_csv(ucf_dataset_valid_path_task0)

task_0_labels = sorted(ucf_train_task0['label'].unique().tolist())
task_1_labels = sorted(train_ucf_task1['label'].unique().tolist())

all_labels = task_0_labels + task_1_labels

id2label = {}
label2id = {}
for index, label in enumerate(all_labels):
    id2label[index] = label
    label2id[label] = index
    
for label in task_0_labels:
    label_df = ucf_train_task0[ucf_train_task0['label'] == label]
    label_df = label_df.sample(random_state=42)
    train_ucf_task1 = pd.concat([train_ucf_task1, label_df])
    
    label_df = ucf_valid_task0[ucf_valid_task0['label'] == label]
    label_df = label_df.sample(random_state=42)
    valid_ucf_task1 = pd.concat([valid_ucf_task1, label_df])
    
train_ucf_task1 = train_ucf_task1.reset_index(drop=True)
valid_ucf_task1 = valid_ucf_task1.reset_index(drop=True)
    
num_old_task_classes = len(task_0_labels)
num_new_task_classes = len(task_1_labels)

class oldCSTAConfig:
    """
    Load the initial model with the EXACT SAME CONFIGS as the previous model.
    In this case the initial model had 50 num_classes
    """
    num_frames = 8                      # taking a lower frame numbers for initial training
    img_size = 224                      # the frames are sized at 256*256
    patch_size = 16                     # patch size
    dim = 480                           # model dimension
    num_classes = num_old_task_classes  # lets say we have a data for initial training with these classes
    num_layers= 12                      # total number of timesformer layers or blocks
    num_channels = 3                    # RGB
    num_heads = 8                       # using 8 heads in attention
    init_with_adapters = True           # for task 0, the model is initialized with one adapter per block
    calculate_distil_loss = False       # For task 0 training, no distillation loss is needed
    calculate_lt_lss_loss = False       # For task 0 training, no lt ls loss is needed
    miu_d = 0.1                         # distillation loss weight
    miu_t = 0.1                         # lt loss weight (currently not implemented)
    miu_s = 0.1                         # ls loss weight (currently not implemented)
    lambda_1 = 0.2                      # new classifiers multiplying factor
    K = 5
    temporal_relations_path = "DATA/UCF101/tasks/task_1/temporal_relations.json"
    spatial_relations_path = "DATA/UCF101/tasks/task_1/spatial_relations.json"
    state_dict_path = "Outputs/Models/Trial_51_run2/best_model.pth"

class DatasetConfig:
    img_size = oldCSTAConfig.img_size
    num_frames = oldCSTAConfig.num_frames
    root_path = "DATA/UCF101"
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    id2label = id2label
    label2id = label2id
    num_labels = len(id2label)

class TrainingConfigs:
    random_seed = 42
    num_training_epochs = 80
    training_batch_size = 6
    evaluation_batch_size = 6
    dataloader_num_workers = 4
    dataloader_pin_memory = False
    dataloader_persistent_workers = False
    learning_rate = 1e-5
    adamw_betas = (0.9, 0.999)
    weight_decay = 1e-5
    eta_min = 1e-10
    T_max = num_training_epochs // 4
    model_output_dir = "Outputs/Models/Task1/Trial1"

def main():
    set_all_seeds(TrainingConfigs.random_seed)
    train_dataset = VideoDataset(train_ucf_task1, 
                                 img_size = DatasetConfig.img_size, 
                                 mean = DatasetConfig.mean, 
                                 std = DatasetConfig.std,
                                 num_frames = DatasetConfig.num_frames,
                                 root_path = DatasetConfig.root_path,
                                 id2label = DatasetConfig.id2label,
                                 label2id = DatasetConfig.label2id,
                                 )
    valid_dataset = VideoDataset(valid_ucf_task1,
                                 img_size = DatasetConfig.img_size, 
                                 mean = DatasetConfig.mean, 
                                 std = DatasetConfig.std,
                                 num_frames = DatasetConfig.num_frames,
                                 root_path = DatasetConfig.root_path,
                                 id2label = DatasetConfig.id2label,
                                 label2id = DatasetConfig.label2id,
                                 )
    
    train_dataloader = DataLoader(train_dataset, 
                                batch_size=TrainingConfigs.training_batch_size, 
                                shuffle=True, 
                                pin_memory=TrainingConfigs.dataloader_pin_memory, 
                                persistent_workers=TrainingConfigs.dataloader_persistent_workers,
                                num_workers=TrainingConfigs.dataloader_num_workers,
                                )
    valid_dataloader = DataLoader(valid_dataset, 
                                batch_size=TrainingConfigs.evaluation_batch_size, 
                                shuffle=False,
                                pin_memory=TrainingConfigs.dataloader_pin_memory, 
                                persistent_workers=TrainingConfigs.dataloader_persistent_workers,
                                num_workers=TrainingConfigs.dataloader_num_workers,
                                )
    
    logging.info(f"\n\nFinetuning starting on: {datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}\n\n")
    # For task n, load the model with the config from model(n-1) 
    # you must have n adapters per blocks
    # distil, ls, and lt losses calculations are set to True
    model = CSTA(**vars(oldCSTAConfig))
    model.load_state_dict(torch.load(oldCSTAConfig.state_dict_path))
    
    # add new task components with new classifier of num_labels size
    model.add_new_task_components(num_old_task_classes + num_new_task_classes)
    model.freeze_all_but_last()
    
    att = model.model_attributes
    print("-"*50)
    logging.info("-"*50)
    for value in att:
        print(f"{value} : {att[value]}")
        logging.info(f"{value} : {att[value]}")
    print(f"Calculate distill loss: {model.calculate_distil_loss}")
    print(f"Calculate lt ls loss: {model.calculate_lt_ls_loss}")
    print(f"Number of classes for this task: {model.classifiers[-1].out_features}")
    print("-"*50)
    logging.info("-"*50)
    
    # optimizer = optim.SGD(model.parameters(), lr = TrainingConfigs.learning_rate, weight_decay=TrainingConfigs.weight_decay)
    optimizer = optim.AdamW(model.parameters(), lr = TrainingConfigs.learning_rate, betas = TrainingConfigs.adamw_betas, weight_decay=TrainingConfigs.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=TrainingConfigs.T_max, eta_min=TrainingConfigs.eta_min)
    
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, valid_dataloader, scheduler
        )
    
    if not os.path.exists(TrainingConfigs.model_output_dir):
        os.makedirs(TrainingConfigs.model_output_dir, exist_ok=True)
    
    best_loss = float('inf')
    best_acc = 0.0
    for epoch in range(TrainingConfigs.num_training_epochs):
        train_loss, _ = train_epoch(model, train_dataloader, optimizer, accelerator, epoch)
        eval_loss, eval_acc = evaluate(model, eval_dataloader, accelerator, epoch)
        
        if eval_loss < best_loss:
            best_loss = eval_loss
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(TrainingConfigs.model_output_dir, 'best_model.pth'))
        elif eval_acc > best_acc:
            best_acc = eval_acc
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(TrainingConfigs.model_output_dir, 'best_model.pth'))
        
        # scheduler.step()
    accelerator.end_training()
    
if __name__ == "__main__":
    filterwarnings("ignore")
    main()