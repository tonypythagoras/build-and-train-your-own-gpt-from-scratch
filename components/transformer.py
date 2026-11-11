import torch
import torch.nn as nn
from layer_norm import LayerNorm
from attention import MultiHeadAttention
from ffn import FFN

class TransfomerBlock(nn.Module):
       def __init__(self,config):
              super().__init__()
              self.layer_norm1=LayerNorm(config)
              self.layer_norm2=LayerNorm(config)
              self.dropout=nn.Dropout(config["drop_rate"])
              self.multi_head_att=MultiHeadAttention(config)
              self.ffn=FFN(config)


       def forward(self,x):
              residual1=x
              x=self.layer_norm1(x)
              x=self.multi_head_att(x)
              x=self.dropout(x)
              x=x+residual1

              residual2=x

              x=self.layer_norm2(x)
              x=self.ffn(x)
              x=self.dropout(x)
              output=x+residual2
              return output