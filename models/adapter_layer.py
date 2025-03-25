import torch
import torch.nn as nn

class ContinualAdapterLayer(nn.Module):
  def __init__(self, in_dim:int, hidden_dim:int):
    super().__init__()
    self.in_dim = in_dim
    self.hidden_dim = hidden_dim

    self.down_projections = nn.ParameterList()
    self.up_projections = nn.ParameterList()

    self.relu = nn.ReLU()
    self.current_task = None

  def add_new_task(self):
    for i in range(len(self.down_projections)):
      self.down_projections[i].requires_grad = False
      self.up_projections[i].requires_grad = False

    down = nn.Parameter(torch.randn(self.in_dim, self.hidden_dim))
    up = nn.Parameter(torch.randn(self.hidden_dim, self.in_dim))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.down_projections.append(down).to(device)
    self.up_projections.append(up).to(device)

    new_task_id = len(self.down_projections)-1
    self.current_task = new_task_id

    self.down_projections[new_task_id].requires_grad = True
    self.up_projections[new_task_id].requires_grad = True

  def set_current_task(self, task_id:int):
    self.current_task = task_id

    for i in range(len(self.down_projections)):
      is_trainable = (i == task_id)
      self.down_projections[i].requires_grad = is_trainable
      self.up_projections[i].requires_grad = is_trainable

  def forward(self, x):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if self.current_task is None:
      raise ValueError("No task has been added yet. add_new_task() first")
    
    if x.dim() == 3:
      B, S, D = x.shape
      T = len(self.down_projections)
      x = x.unsqueeze(1).expand(-1, T, -1, -1)
    elif x.dim() == 4:
      T = len(self.down_projections)

    else:
      raise ValueError(f"unexpected input shape : {x.shape}")
    
    W_dp = torch.stack(self.down_projections, dim = 0).to(device)      #(T, D, d)
    W_up = torch.stack(self.up_projections, dim = 0).to(device)     #(T, d, D)

    out = torch.einsum('btsd, tdh -> btsh', x, W_dp)
    out = self.relu(out)
    out = torch.einsum('btsh, thd -> btsd', out, W_up)
  
    return out