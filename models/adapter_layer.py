import torch
import torch.nn as nn

import torch

def QR_init(
    new_param: torch.nn.Parameter,
    old_params: nn.ParameterList
):
  with torch.no_grad():
    print(new_param.shape[1], new_param.shape[0])
    is_down = new_param.shape[1] < new_param.shape[0]    
    if not old_params:
      if is_down:
        Q, _ = torch.linalg.qr(torch.randn(*new_param.shape, device = new_param.device), mode = 'reduced')
        new_param.data = Q
      else:
        Q, _ = torch.linalg.qr(torch.randn(*new_param.shape, device=new_param.device).T, mode = 'reduced')
        new_param.data = Q.T
      return

    if is_down:      # down
      V = torch.cat([p for p in old_params], dim = 1)
      R = torch.randn_like(new_param)
      proj = V @ torch.linalg.solve(V.T @ V, V.T @ R)
      R_orth = R - proj

      Q, _ = torch.linalg.qr(R_orth)
      new_param.data = Q

    else:     # up
      U = torch.cat([p for p in old_params], dim = 0)
      R = torch.randn_like(new_param)
      proj = (R @ U.T) @ torch.linalg.solve(U @ U.T, U)
      R_orth = R - proj

      R_t = R_orth.T
      Q_t, _ = torch.linalg.qr(R_t, mode = 'reduced')
      Q = Q_t.T
      new_param.data = Q

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

    # print(f"length down before adding weight : {len(self.down_projections)}")
    # print(f"length up before adding weight : {len(self.up_projections)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    down = nn.Parameter(torch.empty(self.in_dim, self.hidden_dim)).to(device)
    up = nn.Parameter(torch.empty(self.hidden_dim, self.in_dim)).to(device)

    QR_init(down, self.down_projections)
    QR_init(up, self.up_projections)

    self.down_projections.append(down).to(device)
    self.up_projections.append(up).to(device)

    new_task_id = len(self.down_projections)-1
    self.current_task = new_task_id

    self.down_projections[new_task_id].requires_grad = True
    self.up_projections[new_task_id].requires_grad = True

    # print(f"length down before adding weight : {len(self.down_projections)}")

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
    
    W_dp_list = [p for p in self.down_projections]
    W_up_list = [p for p in self.up_projections]
    
    W_dp = torch.cat(W_dp_list, dim = 1).to(device)      #(D, t*d)
    W_up = torch.cat(W_up_list, dim = 0).to(device)     #(t*d, D)

    out = self.relu(x @ W_dp) @ W_up
  
    return out