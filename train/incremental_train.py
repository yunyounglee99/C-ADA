import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..models.orthogonal_loss import orthogonal_loss
from ..models.vit import ViTBlockWithCADA, CADA_ViTModel

# 이 코드 내에서 .extend를 쓰는게 맞는지 생각해보기 각 S&S, CAL 블록별로 Ortho loss를 구해야하지 않을까? : 순서만 맞게한다면 상관없을듯

def extract_task_weight_block(block, task_id):
  w_dp_mhsa = block.cal_msha.down_projections[task_id].weight
  w_up_mhsa = block.cal_msha.up_projections[task_id].weight
  w_dp_mlp = block.cal_mlp.down_projections[task_id].weight
  w_up_mlp = block.cal_mlp.up_projections[task_id].weight

  dp_list = [w_dp_mhsa, w_dp_mlp]
  up_list = [w_up_mhsa, w_up_mlp]

  return dp_list, up_list

def extract_task_weight(model, task_id):
  dp_list  = []
  up_list = []
  for block in model.base_vit.encoder.layer:
    if block.cadablock:
      dps, ups = extract_task_weight_block(block, task_id)
      dp_list.extend(dps)
      up_list.extend(ups)
    else:
      pass

  return dp_list, up_list

def train(model, train_loader, task_id, num_epochs, lr, alpha):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  model.add_new_task()
  if task_id > 0:
    model.freeze_sns()

  optimizer = optim.Adam(model.parameters(), lr)
  ce_loss_fn = nn.CrossEntropyLoss()

  old_dp_list = []
  old_up_list = []

  for old_id in range(task_id):
    dp_list, up_list = extract_task_weight(model, old_id)
    old_dp_list.extend(dp_list)
    old_up_list.extend(up_list)

  model.train()
  for epoch in range(num_epochs):
    total_loss = 0.0
    for images, labels in train_loader:
      images, labels = images.to(device), labels.to(device)

      optimizer.zero_grad()
      logits = model(images)
      ce_loss = ce_loss_fn(logits, labels)

      ortho_val = torch.tensor(0.0).to(device)
      if task_id > 0:
        new_dp_list, new_up_list = extract_task_weight(model, task_id)
        for w_dp in new_dp_list:
          for w_up in new_up_list:
            ortho_val += orthogonal_loss(w_dp, w_up, old_dp_list, old_up_list)

      total_loss_val = ce_loss + alpha * ortho_val
      total_loss_val.backward()
      optimizer.step()

      total_loss += total_loss_val.item()
    avg_loss = total_loss / len(train_loader)
    print(f"[task {task_id}] epoch {epoch+1}/{num_epochs}, loss = {avg_loss : .4f}")