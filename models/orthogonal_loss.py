import torch
import math

def orthogonal_loss(
    w_t_dp : torch.Tensor,
    w_t_up : torch.Tensor,
    old_dp_list : list[torch.Tensor],
    old_up_list : list[torch.Tensor]
):
  # w_t_dp, w_t_up도 gpu로 들어가게 하기
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
  loss_dp = torch.tensor(0.0).to(device)
  loss_up = torch.tensor(0.0).to(device)

  if len(old_dp_list) > 0:
    old_dp_cat = torch.cat(old_dp_list, dim = 1)
    dp_mat = w_t_dp.transpose(0, 1) @ old_dp_cat
    loss_dp = torch.norm(dp_mat)**2

  if len(old_up_list) > 0:
    old_up_cat = torch.cat(old_up_list, dim = 0)
    up_mat = w_t_up @ old_up_cat.transpose(0,1)
    loss_up = torch.norm(up_mat)**2
  loss = loss_dp + loss_up

  return loss