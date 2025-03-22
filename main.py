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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

model = CADA_ViTModel(
  hidden_dim = hidden_dim,
)