import torch
import torch.nn as nn

class FFN(nn.Module):
       def __init__(self, config):
              super().__init__()
              self.seq=nn.Sequential(
                     nn.Linear(config["embed_dim"],4*config["embed_dim"]),
                     nn.GELU(),
                     nn.Linear(4*config["embed_dim"],config["embed_dim"]))

       def forward(self,x):
           return self.seq(x)   