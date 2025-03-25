import torch
import yaml
import random
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision 
import torchvision.transforms as transforms

from models.orthogonal_loss import orthogonal_loss
from models.vit import CADA_ViTModel, ViTBlockWithCADA
from train.incremental_train import train_incremental, create_task_dataloaders, evaluate_task

def load_config(config_path):
  with open(config_path, "r") as f:
    return yaml.safe_load(f)
  
parser = argparse.ArgumentParser()
parser.add_argument("--config", type = str, default = "config.yaml")
args = parser.parse_args()
config = load_config(args.config)

all_classes = list(range(100))
# random.shuffle(all_classes)

num_tasks = 10
num_classes = 10

task_splits = []
for i in range(num_tasks):
  start = i * num_classes
  end = (i+1) * num_classes
  task_classes = all_classes[start:end]
  task_splits.append(task_classes)

transform_train = transforms.Compose([
  transforms.Resize((224,224)),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
transform_test = transforms.Compose([
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

train_loaders, test_loaders = create_task_dataloaders(
  train_dataset,
  test_dataset,
  task_splits,
  batch_size = config['batch_size']
)

model = CADA_ViTModel(
  hidden_dim_msha = config['hidden_dim_msha'],
  hidden_dim_mlp = config['hidden_dim_mlp'],
  num_classes = num_classes,
  model_name="google/vit-base-patch16-224-in21k"
)

for task_id in range(num_tasks):
  print(f"\n=== Train task {task_id}=====================")
  train_loader_i =train_loaders[task_id]
  train_incremental(
    model = model,
    train_loader = train_loader_i,
    task_id = task_id,
    num_epochs = config['num_epochs'],
    lr = config['lr'],
    delta = config['delta']
  )

  test_loader_i = test_loaders[task_id]
  acc = evaluate_task(model, test_loader_i)
  print(f"Task {task_id} Test Acc : {acc:.2f}%")