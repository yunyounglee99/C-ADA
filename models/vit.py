import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel

from models.adapter_layer import ContinualAdapterLayer
from models.scaling_and_shifting import ScailingAndShifting

class ViTBlockWithCADA(nn.Module):
  def __init__(self, original_block, embed_dim, hidden_dim):
    super().__init__()
    self.original_block = original_block
    self.cal_msha = ContinualAdapterLayer(embed_dim, hidden_dim)
    self.cal_mlp = ContinualAdapterLayer(embed_dim, hidden_dim)
    self.lamda = 0.1    # 일부러 스펠링 틀린것
    self.sns = ScailingAndShifting(embed_dim)
    self.sns_frozen = False

  def add_new_task(self):
    self.cal_msha.add_new_task()
    self.cal_mlp.add_new_task()

  def set_current_task(self, task_id):
    self.cal_msha.set_current_task(task_id)
    self.cal_mlp.set_current_task(task_id)

  def freeze_sns(self):
    for param in self.sns.parameters():
      param.requires_grad = False

  def forward(self, hidden_states, attention_mask = None):
    # 1. layer norm
    hidden_states = self.original_block.layernorm_before(hidden_states)
    
    # 2. attention block
    sa_out = self.original_block.attention.attention(
      hidden_states,
      head_mask = None,
      output_attentions = False
    )
    sa_out = sa_out[0]

    # 3. S&S, CAL
    sns_out1 = self.sns(sa_out)
    cal1_out = self.cal_msha(sns_out1)

    x_l1 = self.original_block.attention.output.dense(sa_out)
    x_l2 = self.lamda * cal1_out
    x_prime = x_l1 + x_l2

    # 4. intermediate (IDK exact structure of C-ADA but I just fallow the official ViT's structure)
    x_prime = hidden_states + x_prime
    x_prime = self.original_block.layernorm_after(x_prime)

    # 5. MLP part
    mlp_out = self.original_block.intermediate(x_prime)
    mlp_out = self.original_block.output(mlp_out, x_prime)
    sns_out2 = self.sns(x_prime)
    cal2_out = self.cal_mlp(sns_out2)
    x_out = mlp_out + self.lamda * cal2_out

    return x_out

class CADA_ViTModel(nn.Module):
  def __init__(self,
              hidden_dim,
              num_classes,
              model_name = "google/vit-base-patch16-224-in21k"):
    super().__init__()
    self.config = ViTConfig.from_pretrained(model_name)
    embed_dim = self.config.hidden_size

    self.base_vit = ViTModel.from_pretrained(model_name)

    for param in self.base_vit.parameters():
      param.requires_grad = False     # freeze backbone

    for i, block in enumerate(self.base_vit.encoder.layer):
      wrapped_block = ViTBlockWithCADA(
        original_block=block,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
      )
      self.base_vit.encoder.layer[i] = wrapped_block

    self.classifier = nn.Linear(embed_dim, num_classes)
    self.current_task = None

  def add_new_task(self):
    for block in self.base_vit.encoder.layer:
      block.add_new_task()

  def set_current_task(self, task_id):
    self.current_task = task_id
    for block in self.base_vit.encoder.layer:
      block.set_current_task(task_id)
      
  def freeze_sns(self):
    for block in self.base_vit.encoder.layer:
      block.freeze_sns()

  def forward(self, pixel_values, attention_mask = None):
    outputs = self.base_vit(
      pixel_values = pixel_values,
      attention_mask = attention_mask,
      output_attentions = False,
      output_hidden_states = False,
      return_dict = True
    )
    last_hidden_state = outputs.last_hidden_state
    cls_token = last_hidden_state[:,0,:]
    logits = self.classifier(cls_token)

if __name__ == "__main__":
  model_name = "google/vit-base-patch16-224-in21k"
  model = ViTModel.from_pretrained(model_name)
  print(model)
