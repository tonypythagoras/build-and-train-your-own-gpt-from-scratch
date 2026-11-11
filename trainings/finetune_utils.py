from torch.utils.data import Dataset,DataLoader
from functools import partial
import torch

def format(entry):
      instruct=(f"Below is an instruction that describes a task."
                f"Write a response that appropriately completes the request."
                f"\n\n### Instruction:\n{entry["instruction"]}"
                )
      input=(f"\n\n### Input:\n{entry["input"]}" if entry["input"] else "")

      output=instruct+input
      return output

def my_collate_func(batch,pad=50256,maximum=None,ignore=-100):
      input_ls=[]
      target_ls=[]
      longest=max([len(b)+1 for b in batch])
      for entry in batch:
             copy=entry.copy()
             copy +=[pad]

             padded=(copy + [pad]*(longest-len(copy)))
             input_t=padded[:-1]
             target_t=padded[1:]
             input_tensor=torch.tensor(input_t)
             target_tensor=torch.tensor(target_t)
             mask=target_tensor==pad
             indices=torch.nonzero(mask).squeeze()
             if indices.numel()>1:
                    target_tensor[indices[1:]]=ignore

             if maximum is not None:
                    input_tensor=input_tensor[:maximum]
                    target_tensor=target_tensor[:maximum]
             input_ls.append(input_tensor)
             target_ls.append(target_tensor)
      final_input=torch.stack(input_ls)
      final_target=torch.stack(target_ls)
      return final_input,final_target            
      
def get_collate_fn():
       collate_fn=partial(
            my_collate_func,
            maximum=1024,
            
       )
       return collate_fn


def sample(model,idx,max_tok,con_len,temp,topk,ieo_id):
          model.eval()
          for _ in range(max_tok):
                  with torch.no_grad():
                          id=idx[:,-con_len:]
                          logit=model(id)
                          logit=logit[:,-1,:]
                  if topk is not None:
                          top,_=torch.topk(logit,topk)
                          logit=torch.where(
                                  condition=logit < top[:,-1],
                                  input=torch.tensor(float('-inf')),
                                  other=logit
                          )
                  if temp > 0.0:
                          logit=logit/temp
                          prob=torch.softmax(logit,dim=-1)
                          idn=torch.multinomial(prob,num_samples=1)             
                  else:
                          idn=torch.argmax(logit,dim=-1,keepdim=True)
                  if idn == ieo_id:
                          break              

                  idx=torch.cat((idx,idn),dim=1)
          return idx        



def generate_answer(model,tokenizer,con_len,data):
       for entry in data[10:30]:
              instruct=format(entry)
              id_token=text_to_token(instruct,tokenizer)
              idxs=sample(model=model,idx=id_token,max_tok=1024,con_len=con_len,temp=0.0,topk=None,ieo_id=50256)
              gen_text=token_to_text(idxs,tokenizer)  
              response_text=(gen_text[len(instruct):].replace("### Response:","").strip())
              print(f"Quest: {instruct}")
              print(f"Normal response is: {entry["output"]}")
              print(f"Model response is: {response_text.strip()}")
              print("_____________________________________________")  


class MyDataset(Dataset):
      def __init__(self,data,tokenizer):
            self.data=data
            self.tokens=[]
            for entry in data:
                  instruct=format(entry)
                  response=f"\n\n### Response:\n{entry["output"]}"
                  full_text=instruct+response
                  token=tokenizer.encode(full_text)
                  self.tokens.append(token)
                  
      def __getitem__(self, index):
            return self.tokens[index]
      

      def __len__(self):
            return len(self.data)


def text_to_token(text,tokenizer):
       ids=tokenizer.encode(text,allowed_special={'<|endoftext|>'})
       b_ids=torch.tensor(ids).unsqueeze(0)
       return b_ids


def token_to_text(idx,tokenizer):
       ids=idx.squeeze(0)
       text=tokenizer.decode(ids.tolist())
       return text