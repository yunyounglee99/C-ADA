import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel

from models.adapter_layer import ContinualAdapterLayer
from models.scaling_and_shifting import ScailingAndShifting

class ViTBlockWithCADA(nn.Module):
  def __init__(self, original_block, embed_dim, hidden_dim_msha, hidden_dim_mlp, cadablock = True):
    super().__init__()
    self.original_block = original_block
    self.cal_msha = ContinualAdapterLayer(embed_dim, hidden_dim_msha)
    self.cal_mlp = ContinualAdapterLayer(embed_dim, hidden_dim_mlp)
    self.lamda = 0.1    # 일부러 스펠링 틀린것
    self.sns = ScailingAndShifting(embed_dim)
    self.sns_frozen = False
    self.cadablock = cadablock

  def add_new_task(self):
    if not self.cadablock:
      return
    
    self.cal_msha.add_new_task()
    self.cal_mlp.add_new_task()


  def set_current_task(self, task_id):
    if not self.cadablock:
      return
    self.cal_msha.set_current_task(task_id)
    self.cal_mlp.set_current_task(task_id)

  def freeze_sns(self):
    if not self.cadablock:
      return
    
    for param in self.sns.parameters():
      param.requires_grad = False

  def forward(self, hidden_states, head_mask=None, output_attention=False):
    if not self.cadablock:
      out = self.original_block.forward(
        hidden_states,
        head_mask = head_mask,
        output_attentions=output_attention
      )[0]
      return (out,)

    # 1. layer norm
    hidden_states = self.original_block.layernorm_before(hidden_states)
    
    # 2. attention block
    sa_out = self.original_block.attention.attention(
      hidden_states.contiguous(),
      head_mask = None,
      output_attentions = False
    )
    sa_out = sa_out[0]

    # 3. S&S, CAL
    sns_out1 = self.sns(sa_out)
    cal1_out = self.cal_msha(sns_out1)

    x_l1 = self.original_block.attention.output(sa_out, hidden_states)
    x_l2 = self.lamda * cal1_out
    x_prime = x_l1 + x_l2

    # 4. intermediate (IDK the exact structure of C-ADA but I just fallowed the official ViT's structure)
    # x_prime = hidden_states + x_prime     # 1st residual connection
    x_prime = self.original_block.layernorm_after(x_prime)

    # 5. MLP part
    mlp_out = self.original_block.intermediate(x_prime)
    mlp_out = self.original_block.output.dense(mlp_out)     # 2nd residual connection
    mlp_out = self.original_block.output.dropout(mlp_out)

    sns_out2 = self.sns(x_prime)
    cal2_out = self.cal_mlp(sns_out2)
    x_out = mlp_out + self.lamda * cal2_out


    return (x_out,)

class CADA_ViTModel(nn.Module):
  def __init__(self,
              hidden_dim_msha,
              hidden_dim_mlp,
              num_classes,
              model_name = "google/vit-base-patch16-224-in21k"):
    super().__init__()
    self.config = ViTConfig.from_pretrained(model_name)
    embed_dim = self.config.hidden_size

    self.base_vit = ViTModel.from_pretrained(model_name)

    for param in self.base_vit.parameters():
      param.requires_grad = False     # freeze backbone

    self.cadablock = False
    for i, block in enumerate(self.base_vit.encoder.layer):
      if i < 5:
        wrapped_block = ViTBlockWithCADA(
          original_block=block,
          embed_dim=embed_dim,
          hidden_dim_msha=hidden_dim_msha,
          hidden_dim_mlp=hidden_dim_mlp,
          cadablock=True
        )
        self.base_vit.encoder.layer[i] = wrapped_block
      else:
        wrapped_block = ViTBlockWithCADA(
          original_block=block,
          embed_dim=embed_dim,
          hidden_dim_msha=hidden_dim_msha,
          hidden_dim_mlp=hidden_dim_mlp,
          cadablock=False
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

  def forward(self, pixel_values):
    outputs = self.base_vit(
      pixel_values = pixel_values,
      output_attentions = False,
      output_hidden_states = False,
      return_dict = True
    )
    output = outputs.pooler_output
    logits = self.classifier(output)
    return logits

'''
if __name__ == "__main__":
  model_name = "google/vit-base-patch16-224-in21k"
  model = ViTModel.from_pretrained(model_name)
  print(model)

'''