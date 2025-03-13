import torch
import torch.nn as nn

class ScailingAndShifting(nn.Module):
  def __init__(self, embed_dim):
    super().__init__()
    self.alpha = nn.Parameter(torch.ones(embed_dim))
    self.beta = nn.Parameter(torch.zeros(embed_dim))

  def forward(self, x):
    out = self.alpha * x + self.beta
    return out