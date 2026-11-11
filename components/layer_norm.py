import torch
import torch.nn as nn


class LayerNorm(nn.Module):
       def __init__(self,config):
              super().__init__()
              self.eps=1e-5
              self.scale=nn.Parameter(torch.ones(config["embed_dim"]))
              self.shift=nn.Parameter(torch.zeros(config["embed_dim"]))

       def forward(self,x):
              mean=x.mean(dim=-1,keepdim=True)
              var=x.var(dim=-1,keepdim=True,unbiased=False)
              x_norm=(x-mean)/torch.sqrt(var+self.eps)
              output=x_norm * self.scale + self.shift
              return output  