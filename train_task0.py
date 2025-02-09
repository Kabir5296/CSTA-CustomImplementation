from src.model import CSTA
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

logging.basicConfig(
    filename="logs/train.log",
    filemode="a",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

ucf_dataset_training_path = "DATA/UCF101/tasks/task_0/train.csv"
ucf_dataset_test_path = "DATA/UCF101/tasks/task_0/test.csv"
ucf_dataset_valid_path = "DATA/UCF101/tasks/task_0/val.csv"

# ucf_dataset_training_path = "DATA/UCF101/tasks/task_beta_0/train.csv"
# ucf_dataset_test_path = "DATA/UCF101/tasks/task_beta_0/test.csv"
# ucf_dataset_valid_path = "DATA/UCF101/tasks/task_beta_0/val.csv"

ucf_train = pd.read_csv(ucf_dataset_training_path)
ucf_valid = pd.read_csv(ucf_dataset_valid_path)

all_labels = sorted(ucf_train['label'].unique().tolist())
id2label = {}
label2id = {}
for index, label in enumerate(all_labels):
    id2label[index] = label
    label2id[label] = index

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
    
class TrainingConfigs:
    random_seed = 42
    num_training_epochs = 80
    training_batch_size = 8
    evaluation_batch_size = 8
    dataloader_num_workers = 4
    dataloader_pin_memory = False
    dataloader_persistent_workers = False
    learning_rate = 1e-5
    adamw_betas = (0.9, 0.999)
    weight_decay = 1e-5
    eta_min = 1e-10
    warmup_epochs = 12
    T_max = num_training_epochs - warmup_epochs
    model_output_dir = "Outputs/Models/Trial_51_run2"

def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
            transforms.Normalize(mean=mean, std=std),
            # transforms.RandomGrayscale(0.1),
            # transforms.RandomRotation((-15,15)),
            # transforms.RandomErasing(0.1),
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
        return video.float()
    
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

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
def check_specific_gradients(model):
    components = {
        'patch_embed': model.patch_embed,
        'temporal_msa': model.blocks[0].temporal_msa,
        'spatial_msa': model.blocks[0].spatial_msa,
        'temporal_adapters': model.temporal_adapters[0],
        'spatial_adapters': model.spatial_adapters[0],
        'classifier': model.classifiers[-1]
    }
    with open("logs/gradients.txt", "+a") as f:
        f.write("New Batch\n\n")
        for name, component in components.items():
            grad_norm = 0
            for p in component.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item()
                    f.write(f"{name} gradient norm: {grad_norm:.6f}\n")

def train_epoch(model, train_dataloader, optimizer, accelerator, epoch):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0
    
    progress_bar = tqdm(
        total=len(train_dataloader),
        disable=not accelerator.is_local_main_process,
        desc=f"Training epoch {epoch}"
    )
    for batch_idx, batch in enumerate(train_dataloader):
        input_frames = batch["input_frames"]
        labels = batch["label"]
        batch_size = labels.size(0)
        
        with accelerator.accumulate(model):
            outputs = model(input_frames, labels)
            loss = outputs.loss
            
            predictions = outputs.predictions.argmax(-1)
            correct = (predictions == labels).sum().item()
            accuracy = correct / batch_size
            # pdb.set_trace()
            accelerator.backward(loss)
            check_specific_gradients(model)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)  # Add gradient clipping
            optimizer.step()
            optimizer.zero_grad()
        
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
    logging.info(
        f"Epoch {epoch} - "
        f"Average Loss: {avg_loss:.4f}, "
        f"Average Accuracy: {avg_acc:.4f}, "
        f"Samples: {total_samples}"
    )
    
    return avg_loss, avg_acc

def evaluate(model, eval_dataloader, accelerator, epoch):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0
    
    progress_bar = tqdm(
        total=len(eval_dataloader),
        disable=not accelerator.is_local_main_process,
        desc=f"Evaluation epoch {epoch}"
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
    logging.info(
        f"Evaluation Epoch {epoch} - "
        f"Average Loss: {avg_loss:.4f}, "
        f"Average Accuracy: {avg_acc:.4f}, "
        f"Samples: {total_samples}"
    )
    
    return avg_loss, avg_acc

def main():
    set_all_seeds(TrainingConfigs.random_seed)
    train_dataset = VideoDataset(ucf_train)
    valid_dataset = VideoDataset(ucf_valid)

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
    logging.info(f"\n\nTraining starting on: {datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}\n\n")
    # For task 0 training, the default configs are fine. Need to check though: 
    # you must have one adapter per blocks
    # distil, ls, and lt losses calculations are set to zero
    model = CSTA(**vars(CSTAConfig))
    att = model.model_attributes
    print("-"*50)
    logging.info("-"*50)
    for value in att:
        print(f"{value} : {att[value]}")
        logging.info(f"{value} : {att[value]}")
    print(f"Calculate distill loss: {model.calculate_distil_loss}")
    print(f"Calculate lt ls loss: {model.calculate_lt_ls_loss}")
    print(f"Number of classes for this task: {CSTAConfig.num_classes}")
    print("-"*50)
    logging.info("-"*50)
    
    # optimizer = optim.SGD(model.parameters(), lr = TrainingConfigs.learning_rate, weight_decay=TrainingConfigs.weight_decay)
    optimizer = optim.AdamW(model.parameters(), lr = TrainingConfigs.learning_rate, betas = TrainingConfigs.adamw_betas, weight_decay=TrainingConfigs.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=TrainingConfigs.T_max, eta_min=TrainingConfigs.eta_min)
    
    model.apply(init_weights)
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
        if epoch > TrainingConfigs.warmup_epochs:
            scheduler.step()
    accelerator.end_training()
    
if __name__ == "__main__":
    filterwarnings("ignore")
    main()