import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel

from models.adapter_layer import ContinualAdapterLayer
from models.scaling_and_shifting import ScailingAndShifting

class ViTBlockWithCADA(nn.Module):
  def __init__(self, original_block, embed_dim, hidden_dim, total_tasks):
    super().__init__()
    self.original_block = original_block
    self.cal_msha = ContinualAdapterLayer(embed_dim, hidden_dim, total_tasks)
    self.cal_mlp = ContinualAdapterLayer(embed_dim, hidden_dim, total_tasks)
    self.lamda = 0.1    # 일부러 스펠링 틀린것
    self.sns = ScailingAndShifting(embed_dim)
    self.sns_frozen = False

  def set_current_task(self, task_id):
    self.cal_msha.set_current_task(task_id)
    self.cal_mlp.set_current_task(task_id)

  def freeze_sns(self):
    self.sns_frozen = True
    for param in self.sns.parameters():
      param.requires_grad = True

  def forward(self, hidden_states, attention_mask = None):
    if not self.sns_frozen:
      hidden_states = self.sns(hidden_states)
      sa_output = self.original_block.attention(     # sa : self attention
        hidden_states,
        attention_mask = attention_mask,
        output_attentions = False
      )
      sa_hidden = sa_output[0]

      cal1_out = self.cal_msha(sa_hidden)
      x_prime = sa_hidden + self.lamda * cal1_out
      x_prime = self.original_block.layernorm_before_mlp(x_prime)

      mlp_out = self.original_block.mlp(x_prime)
      cal2_out = self.cal_mlp(x_prime)
      x_out = mlp_out + self.lamda * cal2_out

if __name__ == "__main__":
  model_name = "google/vit-base-patch16-224-in21k"
  model = ViTModel.from_pretrained(model_name)
  print(model)
