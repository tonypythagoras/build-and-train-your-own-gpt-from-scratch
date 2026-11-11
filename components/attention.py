import torch
import torch.nn as nn
from config import CONFIG

class MultiHeadAttention(nn.Module):
       def __init__(self,config):
              super().__init__()
              self.n_heads=config["n_heads"]
              self.n_out=config["embed_dim"]
              self.head_dim=self.n_out//self.n_heads
              self.dropout=nn.Dropout(config["drop_rate"])
              self.out_proj=nn.Linear(config["embed_dim"],config["embed_dim"])
              self.W_query=nn.Linear(config["embed_dim"],config["embed_dim"])
              self.W_key=nn.Linear(config["embed_dim"],config["embed_dim"])
              self.W_value=nn.Linear(config["embed_dim"],config["embed_dim"])
              self.register_buffer('mask',
                                   torch.triu(torch.ones(config["con_length"],config["con_length"]),
                                   diagonal=1))

       def forward(self,x):
              batch,num_tok,tok_dim=x.shape
              query=self.W_query(x)
              key=self.W_key(x)
              value=self.W_value(x)
              key=key.view(batch, num_tok,self.n_heads,self.head_dim)
              query=query.view(batch, num_tok,self.n_heads,self.head_dim)
              value=value.view(batch, num_tok,self.n_heads,self.head_dim)
              query=query.transpose(1,2)
              key=key.transpose(1,2)
              value=value.transpose(1,2)
              att_score=query @ key.transpose(2,3)
              mask_bool=self.mask.bool()[:num_tok, :num_tok]
              att_score.masked_fill_(mask_bool,-torch.inf)
              att_weigth=torch.softmax(att_score/key.shape[-1]**0.5,dim=-1)
              att_weigth=self.dropout(att_weigth)
              context_vec=att_weigth @ value
              context_vec=context_vec.transpose(1,2)
              context_vec=context_vec.contiguous().view(batch,num_tok,self.n_out)
              output_vec=self.out_proj(context_vec)
              return output_vec