import torch
import torch.nn as nn

class ContinualAdapterLayer(nn.Module):
  def __init__(self, in_dim:int, hidden_dim:int, total_tasks:int):
    super().__init__()
    self.in_dim = in_dim
    self.hidden_dim = hidden_dim
    self.total_tasks = total_tasks

    self.down_projections = nn.ModuleList()
    self.up_projections = nn.ModuleList()

    for _ in range(total_tasks):
      down = nn.Linear(in_dim, hidden_dim, bias = False)
      up = nn.Linear(hidden_dim, in_dim, bias = False)
      self.down_projections.append(down)
      self.up_projections.append(up)

    self.relu = nn.ReLU()
    self.current_task = 0 

  def set_current_task(self, task_id:int):
    self.current_task = task_id

    for i in range(self.total_tasks):
      is_trainable = (i == task_id)
      for parameter in self.down_projections[i].parameters():
        parameter.requires_grad = is_trainable
      for parameter in self.up_projections[i].parameters():
        parameter.requires_grad = is_trainable

  def forward(self, x):
    down_w = self.down_projections[self.current_task]
    up_w = self.up_projections[self.current_task]

    out = self.relu(down_w(x))
    out = up_w(out)
    return out