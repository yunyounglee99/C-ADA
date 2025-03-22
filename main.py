import torch
import yaml
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision 
import torchvision.transforms as transforms

from models.orthogonal_loss import orthogonal_loss
from models.vit import CADA_ViTModel, ViTBlockWithCADA
from train.incremental_train import train_incremental

def load_config(config_path):
  with open(config_path, "r") as f:
    return yaml.safe_load(f)
  
parser = argparse.ArgumentParser()
parser.add_argument("--config", type = str, default = "config.yaml")
args = parser.parse_args()
config = load_config(args.config)

transform_train = transforms.Compose([
  transforms.Resize((224,224)),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
transform_test = transforms([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

train_dataset = torchvision.datasets.CIFAR100(
  root = "./data",
  train = True,
  download = True,
  transform = transform_train
)
test_dataset = torchvision.datasets.CIFAR100(
  root = "./data",
  train = False,
  download = True,
  transform = transform_test
)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

model = CADA_ViTModel(
  hidden_dim = config['middle_dim'],
  num_classes = config['num_classes'],
  model_name="google/vit-base-patch16-224-in21"
)
train_incremental(
  model = model,
  train_loader = train_loader,
  task_id = 1,
  num_epochs = config['num_epochs'],
  lr = config['lr'],
  delta = config['lr']
)