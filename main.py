import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision 
import torchvision.transforms as transforms

from models.orthogonal_loss import orthogonal_loss
from models.vit import CADA_ViTModel, ViTBlockWithCADA

train_dataset = torchvision.datasets.CIFAR100(
  root = "./data",
  train = True,
  download = True,
  transform = 
)