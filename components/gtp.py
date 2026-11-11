import torch
import torch.nn as nn
from layer_norm import LayerNorm
from transformer import TransfomerBlock

class GPT2(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.tok_embed_layer=nn.Embedding(config["vocab_size"],config["embed_dim"])
        self.pos_embed_layer=nn.Embedding(config["con_length"],config["embed_dim"])
        self.dropout=nn.Dropout(config["drop_rate"])
        self.transformer_layer=nn.Sequential(*[TransfomerBlock(config) for _ in range(config["n_heads"])])
        self.final_layer_norm=LayerNorm(config)
        self.final_output_layer=nn.Linear(config["embed_dim"],config["vocab_size"])


    def forward(self,x):
        batch,num_tok=x.shape
        tok_embed=self.tok_embed_layer(x)
        pos_value=torch.arange(num_tok)
        pos_embed=self.pos_embed_layer(pos_value)
        x_embed=tok_embed+pos_embed
        x_token=self.dropout(x_embed)
        x_token=self.transformer_layer(x_token)
        x_output=self.final_layer_norm(x_token)
        logit=self.final_output_layer(x_output)
        return logit
