import torch
from config import CONFIG
import tiktoken
from pretrain_utils import get_data_loader,get_optimizer,save_model,train_model
from components.gtp import GPT2
import os
from datasets import load_dataset
import sys


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tokenizer=tiktoken.get_encoding("gpt2")

    dataset = load_dataset(
        "roneneldan/TinyStories",
        split="train")

# First split: 90% training, 10% temporary
    split_1 = dataset.train_test_split(
        test_size=0.10,
        seed=42)

    train_data = split_1["train"]
    temp_data = split_1["test"]

# Second split: Divide temporary data equally
    split_2 = temp_data.train_test_split(
        test_size=0.50,
        seed=42)

    val_data = split_2["train"]
    test_data = split_2["test"]

    num_epoch=4
    start_text="I love cabbage"
    train_loader=get_data_loader(raw_data=train_data,tokenizer=tokenizer,max_len=16,stride=1,batch_size=1,shuffle=False,num_worker=0,drop_last=True)
    val_loader=get_data_loader(raw_data=val_data,tokenizer=tokenizer,max_len=16,stride=1,batch_size=1,shuffle=False,num_worker=0,drop_last=True)

    model=GPT2(CONFIG)
    optimizer=get_optimizer(model=model)

    if os.path.exists("pretrain_base_model.pth"):
        print("Model is already trained. Exiting")
        sys.exit()               
    else:
        model.train()
        train_model(model=model,device=device,train_loader=train_loader,
                                                       num_epoch=num_epoch,optimizer=optimizer,tokenizer=tokenizer,
                                                       sample_text=start_text)
        save_model(model=model,name="pretrain_base") 

if __name__ == "__main__":
    main()

