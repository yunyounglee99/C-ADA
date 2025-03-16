import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..models.orthogonal_loss import orthogonal_loss
from ..models.vit import ViTBlockWithCADA, CADA_ViTModel

def train_first_phase(model, train_loader, num_epochs, lr):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

  model.set_current_task(0)
  optimizer = optim.Adam(model.parameters(), lr = lr)
  ce_loss_fn = nn.CrossEntropyLoss()

  model.train()
  for epoch in range(num_epochs):
    total_loss = 0.0
    for images, labels in train_loader:
      images, labels = images.to(device), labels.to(device)

      optimizer.zero_grad()
      logits = model(images)
      loss = ce_loss_fn(logits, labels)
      loss.backward()
      optimizer.step()

      total_loss += loss.item()
    print(f"[First Phase] Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
  


