import torch

def orthogonal_loss(
    w_t_dp : torch.Tensor,
    w_t_up : torch.Tensor,
    old_dp_list : list[torch.Tensor],
    old_up_list : list[torch.Tensor]
) -> torch.Tensor:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # w_t_dp, w_t_up도 gpu로 들어가게 하기
  loss_dp = torch.tensor(0.0, device = device)
  loss_up = torch.tensor(0.0, device = device)

  if len(old_dp_list) > 0:
    old_dp_cat = torch.cat(old_dp_list, dim = 1)
    dp_mat = w_t_dp.transpose(0, 1) @ old_dp_cat
    loss_dp = (dp_mat ** 2).sum()

  if len(old_up_list) > 0:
    old_up_cat = torch.cat([w.transpose(0,1) for w in old_up_list], dim = 1)
    up_mat = w_t_up @ old_up_cat
    loss_up = (up_mat ** 2).sum()

  loss = loss_dp + loss_up
  
  return loss