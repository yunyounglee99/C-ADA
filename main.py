import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision 
import torchvision.transforms as transforms

from models.orthogonal_loss import orthogonal_loss
from models.vit import CADA_ViTModel, ViTBlockWithCADA

transform_train = transforms.Compose([
  transforms.Resize((224,224)),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, ))
])

train_dataset = torchvision.datasets.CIFAR100(
  root = "./data",
  train = True,
  download = True,
  transform = 
)