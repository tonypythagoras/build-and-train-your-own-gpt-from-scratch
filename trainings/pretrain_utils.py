import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import math

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



class FFN(nn.Module):
       def __init__(self, config):
              super().__init__()
              self.seq=nn.Sequential(
                     nn.Linear(config["embed_dim"],4*config["embed_dim"]),
                     nn.GELU(),
                     nn.Linear(4*config["embed_dim"],config["embed_dim"]))

       def forward(self,x):
           return self.seq(x)               



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


class MyDataset(Dataset):
       def __init__(self,txt,tokenizer,max_len,stride):
              self.input_ids=[]
              self.target_ids=[]
              self.token_ids=tokenizer.encode(txt)

              for i  in range(0,len(self.token_ids)-max_len,stride):
                     input=self.token_ids[i:i+max_len]
                     target=self.token_ids[i+1:i+max_len+1]
                     self.input_ids.append(torch.tensor(input))
                     self.target_ids.append(torch.tensor(target))

       def __getitem__(self, idx):
               return self.input_ids[idx], self.target_ids[idx]
       def __len__(self):
              return len(self.input_ids)
                      
                     

def get_data_loader(raw_data,tokenizer,max_len,stride,batch_size,shuffle,num_worker,drop_last):
       my_dataset=MyDataset(raw_data,tokenizer,max_len,stride)
       data_loader=DataLoader(
              my_dataset,
              batch_size=batch_size,
              shuffle=shuffle,
              num_workers=num_worker,
              drop_last=drop_last               
       )
       return data_loader


def text_to_token(text,tokenizer):
       ids=tokenizer.encode(text,allowed_special={'<|endoftext|>'})
       b_ids=torch.tensor(ids).unsqueeze(0)
       return b_ids


def token_to_text(idx,tokenizer):
       ids=idx.squeeze(0)
       text=tokenizer.decode(ids.tolist())
       return text

def generate_token(model,idx,max,context_len):
       for _ in range(max):
              ids=idx[:,-context_len:]
              with torch.no_grad():
                     logit=model(ids)
                     logit=logit[:,-1,:]
                     prob=torch.softmax(logit,dim=-1)
                     pred=torch.argmax(prob,dim=-1,keepdim=True)
                     idx=torch.cat((idx,pred),dim=1)

       return idx


def get_batch_loss(model,inputs,target):
       logit=model(inputs)
       loss=torch.nn.functional.cross_entropy(logit.flatten(0,1),target.flatten())
       return loss


def get_loader_loss(model,loader,size):
       total_loss=0
       if len(loader) == 0:
              return int("nan")
       elif size is None:
              size=len(loader)
       else:
              size=min(size,len(loader))

       for i, (inputs,target) in enumerate(loader):
              if i < size:
                     loss=get_batch_loss(model=model,inputs=inputs,target=target)
                     total_loss +=loss.item()

              else:
                     break
       return total_loss/size 

def evaluate_model(model,train_loader,eval_loader,num_batches):
      model.eval()
      with torch.no_grad():
             train_loss=get_loader_loss(model=model,loader=train_loader,size=num_batches) 
             eval_loss=get_loader_loss(model=model,loader=eval_loader,size=num_batches) 
      model.train()
      return train_loss,eval_loss

def print_trained_sample(model,tokenizer,start_sample):
       model.eval()
       con_len=model.pos_embed_layer.weight.shape[0]
       idxs=text_to_token(text=start_sample,tokenizer=tokenizer)
       with torch.no_grad():
              gen_ids=generate_token(model=model,idx=idxs,max=100,context_len=con_len)
       gen_text=token_to_text(idx=gen_ids,tokenizer=tokenizer)
       gen_text=gen_text.replace("\n"," ")
       print(gen_text)
       

def train_model(model,device,train_loader,num_epoch,optimizer,tokenizer,sample_text):
      
      global_step=-1
      token_seen=0
      total_steps=len(train_loader) * num_epoch
      warmup_steps=int(0.2*total_steps)
      peak_lr=optimizer.param_groups[0]["lr"]
      initial_lr=3e-5
      min_lr=1e-6
      lr_inc=(peak_lr-initial_lr)/warmup_steps
      lr=initial_lr

      use_amp = device.type != "cpu"
      scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp else None

      for _ in range(num_epoch):
             for inputs,target in train_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)

                    optimizer.zero_grad()
                    if global_step<warmup_steps:
                           lr=initial_lr+global_step*lr_inc
                    else:
                           progress=(global_step-warmup_steps)/(total_steps-warmup_steps)
                           lr=min_lr+(peak_lr-min_lr) * 0.5*(1 + math.cos(math.pi * progress))

                    for param in optimizer.param_groups:
                           param["lr"]=lr
                    with torch.amp.autocast(device.type, enabled=use_amp):
                      loss=get_batch_loss(model=model,inputs=inputs,target=target)
                    if scaler:
                      scaler.scale(loss).backward()
                    else:
                      loss.backward()

                    if scaler:
                      scaler.unscale_(optimizer)

                    if global_step > warmup_steps:
                           torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=0.1)

                    if scaler:
                      scaler.step(optimizer)
                      scaler.update()
                    else:
                      optimizer.step()
                    global_step +=1
                    token_seen +=inputs.numel()
          
             print_trained_sample(model=model,tokenizer=tokenizer,start_sample=sample_text)


def get_optimizer(model):
    no_grad_param=[]
    grad_param=[]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim()<=1 or "norm" in name.lower() or "bias" in name:
            no_grad_param.append(param)
        else:
            grad_param.append(param)

    return torch.optim.AdamW(
        [
            {"params": grad_param, "weight_decay": 0.1},
            {"params": no_grad_param, "weight_decay": 0.0},
        ],
        lr=3e-4,
        betas=(0.9,0.99),
        eps=1e-8,)

def save_model(model,name):
       path=name+"_model.pth"
       torch.save({"model_state_dict" : model.state_dict()},
       path)
       print("Saved the model")
      


