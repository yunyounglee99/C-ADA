import torch
import torch.nn as nn

class ContinualAdapterLayer(nn.Module):
  def __init__(self, in_dim:int, hidden_dim:int):
    super().__init__()
    self.in_dim = in_dim
    self.hidden_dim = hidden_dim

    self.down_projections = nn.ModuleList()
    self.up_projections = nn.ModuleList()

    self.relu = nn.ReLU()
    self.current_task = None

  def add_new_task(self):
    if self.current_task == None:
      pass
    else:
      for i in range(len(self.down_projections)):
        for param in self.down_projections[i].parameters():
          param.requires_grad = False
        for param in self.up_projections[i].parameters():
          param.requires_grad = False

    down = nn.Linear(self.in_dim, self.hidden_dim, bias = False)
    up = nn.Linear(self.hidden_dim, self.in_dim, bias = False)

    self.down_projections.append(down)
    self.up_projections.append(up)

    new_task_id = len(self.down_projections)-1
    self.current_task = new_task_id

    for param in self.down_projections[new_task_id].parameters():
      param.requires_grad = True
    for param in self.up_projections[new_task_id].parameters():
      param.requires_grad = True

  def forward(self, x):
    if self.current_task is None:
      raise ValueError("No task has been added yet. add_new_task() first")
    
    t = len(self.down_projections)
    out = []
    for i in range(t):
      out_i = self.down_projections[i](x)
      out_i = self.relu(out_i)
      out_i = self.up_projections[i](x)
      out.append(out_i)

    out = torch.stack(out, dim = 0)
    out = out.transpose(0, 1)     #(batch, t, seq_len, in_dim)
    return out