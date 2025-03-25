import torch
import torch.nn as nn

def QR_init(new_param:nn.Parameter, old_params:list[nn.Parameter]):
  with torch.no_grad():
    if len(old_params) == 0:
      return
    old_list = [p.data for p in old_params]
    M = torch.cat(old_list, dim = 1)

    Q, R = torch.linalg.qr(M, mode = 'reduced')
    W_new = new_param.data
    proj = Q.T @ W_new
    W_orth = W_new - Q @ proj

    Q2, R2 = torch.linalg.qr(W_orth, model = 'reduced')
    W_final = Q2[:, :W_orth.size(1)]

    new_param.data.copy_(W_final)


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
    
    down = nn.Parameter(torch.randn(self.in_dim, self.hidden_dim))
    up = nn.Parameter(torch.randn(self.hidden_dim, self.in_dim))

    QR_init(down, self.down_projections)
    QR_init(up, self.up_projections)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.down_projections.append(down).to(device)
    self.up_projections.append(up).to(device)

    new_task_id = len(self.down_projections)-1
    self.current_task = new_task_id

    self.down_projections[new_task_id].requires_grad = True
    self.up_projections[new_task_id].requires_grad = True

    # print(f"length down before adding weight : {len(self.down_projections)}")
    # print(f"length up before adding weight : {len(self.up_projections)}")

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