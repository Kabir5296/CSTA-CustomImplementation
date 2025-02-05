from src import CSTA
import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from abc import abstractmethod
from torchvision.io import read_video
import torchvision.transforms as transforms

ucf_dataset_training_path = "DATA/UCF101/train.csv"
ucf_dataset_test_path = "DATA/UCF101/test.csv"
ucf_dataset_valid_path = "DATA/UCF101/val.csv"

ucf_train = pd.read_csv(ucf_dataset_training_path)

all_labels = sorted(ucf_train['label'].unique().tolist())
id2label = {}
label2id = {}
for index, label in enumerate(all_labels):
    id2label[index] = label
    label2id[label] = index

class CSTAConfig:
    num_frames = 8                 # taking a lower frame numbers for initial training
    img_size = 256                 # the frames are sized at 256*256
    patch_size = 16                # patch size
    dim = 768                      # model dimension
    num_classes = len(all_labels)  # lets say we have a data for initial training with these classes
    num_layers= 12                 # total number of timesformer layers or blocks
    num_channels = 3               # RGB
    num_heads = 8                  # using 8 heads in attention
    init_with_adapters = True      # for task 0, the model is initialized with one adapter per block
    calculate_distil_loss = False  # For task 0 training, no distillation loss is needed
    calculate_lt_lss_loss = False  # For task 0 training, no lt ls loss is needed
    miu_d = 1.0                    # distillation loss weight
    miu_t = 1.0                    # lt loss weight (currently not implemented)
    miu_s = 1.0                    # ls loss weight (currently not implemented)

class DatasetConfig:
    img_size = CSTAConfig.img_size
    num_frames = CSTAConfig.num_frames
    root_path = "DATA/UCF101"
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    id2label = id2label
    label2id = label2id
    
class TrainingConfigs:
    training_batch_size = 10
    evaluation_batch_size = 10
    dataloader_num_workers = 4
    dataloader_pin_memory = False
    dataloader_persistent_workers = False

class VideoDataset(Dataset):
    def __init__(self, 
                 df, 
                 img_size = DatasetConfig.img_size, 
                 mean = DatasetConfig.mean, 
                 std = DatasetConfig.std,
                 num_frames = DatasetConfig.num_frames,
                 root_path = DatasetConfig.root_path,
                 id2label = DatasetConfig.id2label,
                 label2id = DatasetConfig.label2id,
                 ):
        self.df = df
        self.num_frames = num_frames
        self.label2id = label2id
        self.id2label = id2label
        self.root_path = root_path
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=mean, std=std)
            ])

    def __len__(self):
        return len(self.df)
    
    @staticmethod
    def sample_frames(video, num_frames):
        total_frames = video.shape[0]
        indices = torch.linspace(0, total_frames - 1, num_frames)
        indices = torch.clamp(indices, 0, total_frames - 1).long()
        frames = video[indices]
        frames = frames.float() / 255.0
        return frames

    @staticmethod
    def load_video(path):
        video, _, _ = read_video(filename=path, output_format="TCHW")
        return video
    
    def __getitem__(self, index):
        video_path = os.path.join(self.root_path, self.df['clip_path'][index][1:])
        label = self.df['label'][index]

        frames = self.sample_frames(self.load_video(video_path), num_frames=self.num_frames)
        processed_frames = torch.stack([
            self.transform(frame) for frame in frames
        ])

        return {
            "input_frames": processed_frames,
            "label": self.label2id[label],
        }
    
train_dataset = VideoDataset(ucf_train)
train_dataloader = DataLoader(train_dataset, 
                              batch_size=TrainingConfigs.training_batch_size, 
                              shuffle=True, 
                              pin_memory=TrainingConfigs.dataloader_pin_memory, 
                              persistent_workers=TrainingConfigs.dataloader_persistent_workers,
                              num_workers=TrainingConfigs.dataloader_num_workers,
                              )
model = CSTA(**vars(CSTAConfig))