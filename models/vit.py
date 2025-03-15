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
    self.lamda = 0.1
    self_sns = ScailingAndShifting(embed_dim)
    self.sns_frozen = False