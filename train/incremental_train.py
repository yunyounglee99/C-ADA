import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from models.orthogonal_loss import orthogonal_loss
from models.vit import ViTBlockWithCADA, CADA_ViTModel
from tqdm import tqdm

# 이 코드 내에서 .extend를 쓰는게 맞는지 생각해보기 각 S&S, CAL 블록별로 Ortho loss를 구해야하지 않을까? : 순서만 맞게한다면 상관없을듯
def extract_task_weight_block(block, task_id):
  w_dp_mhsa = block.cal_msha.down_projections[task_id].weight
  w_up_mhsa = block.cal_msha.up_projections[task_id].weight
  w_dp_mlp = block.cal_mlp.down_projections[task_id].weight
  w_up_mlp = block.cal_mlp.up_projections[task_id].weight

  dp_list = [w_dp_mhsa, w_dp_mlp]
  up_list = [w_up_mhsa, w_up_mlp]

  return dp_list, up_list

def extract_task_weight_model(model, task_id):
  dp_list  = []
  up_list = []
  for block in model.base_vit.encoder.layer:
    dps, ups = extract_task_weight_block(block, task_id)
    dp_list.extend(dps)
    up_list.extend(ups)
  
  return dp_list, up_list

def train_incremental(model, train_loader, task_id, num_epochs, lr, delta):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(device)
  model.to(device)

  model.add_new_task()
  if task_id > 0:
    model.freeze_sns()

  model.set_current_task(task_id)
  model.freeze_sns()

  optimizer = optim.Adam(model.parameters(), lr)
  ce_loss_fn = nn.CrossEntropyLoss()

  old_dp_list = []
  old_up_list = []

  for old_id in range(task_id):
    dp_list, up_list = extract_task_weight_model(model, old_id)
    old_dp_list.extend(dp_list)
    old_up_list.extend(up_list)

  model.train()
  for epoch in tqdm(range(num_epochs)):
    total_loss = 0.0
    for images, labels in train_loader:
      images, labels = images.to(device), labels.to(device)

      optimizer.zero_grad()
      logits = model(images)
      ce_loss = ce_loss_fn(logits, labels)

      new_dp_list, new_up_list = extract_task_weight_model(model, task_id)

      ortho_val = torch.tensor(0.0).to(device)
      for w_dp in new_dp_list:
        for w_up in new_up_list:
          ortho_val += orthogonal_loss(w_dp, w_up, old_dp_list, old_up_list)

      total_loss_val = ce_loss + delta * ortho_val
      total_loss_val.backward()
      optimizer.step()

      total_loss += total_loss_val.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

def evaluate_task(model, test_loader):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    for images, labels in test_loader:
      images, labels = images.to(device), labels.to(device)
      logits = model(images)
      preds = logits.argmax(dim=1)
      correct += (preds == labels).sum().item()
      total += labels.size(0)
  acc = 100.0 * correct / total if total > 0 else 0.0
  return acc

def select_class_subset(dataset, class_list):
  indices = [i for i, target in enumerate(dataset.targets) if target in class_list]
  return Subset(dataset, indices)

def create_task_dataloaders(train_dataset, test_dataset, task_splits, batch_size):
  train_loaders = []
  test_loaders = []

  for classes in task_splits:
    train_subset = select_class_subset(train_dataset, classes)
    train_loader = DataLoader(train_subset, batch_size, shuffle = True, num_workers=2)

    test_subset = select_class_subset(test_dataset, classes)
    test_loader = DataLoader(test_subset, batch_size, shuffle = False, num_workers=2)

    train_loaders.append(train_loader)
    test_loaders.append(test_loader)

  return train_loaders, test_loaders
