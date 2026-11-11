import torch
import tiktoken
from config import CONFIG
from components.gtp import GPT2
import os
from torch.utils.data import DataLoader
import time
import sys
from datasets import load_dataset,load_from_disk, DatasetDict
from finetune_utils import MyDataset,get_collate_fn
from pretrain_utils import get_optimizer,train_model,save_model


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    tokenizer=tiktoken.get_encoding("gpt2")
    model=GPT2(CONFIG)

    if os.path.exists("pretrain_base_model.pth"):
        model_path=torch.load("pretrain_base_model.pth",map_location=device)
        model.load_state_dict(model_path["model_state_dict"])
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        train_temp = dataset.train_test_split(test_size=0.2,seed=42)
        val_test = train_temp["test"].train_test_split(test_size=0.5,seed=42)
        final_dataset = DatasetDict({"train": train_temp["train"],"validation": val_test["train"],"test": val_test["test"]})
        final_dataset.save_to_disk("alpaca_split")
        dataset = load_from_disk("alpaca_split")
        train_data = dataset["train"]
        val_data = dataset["validation"]
        test_data = dataset["test"]
        train_set=MyDataset(train_data,tokenizer)
        test_set=MyDataset(test_data,tokenizer)
        val_set=MyDataset(val_data,tokenizer)
        collate=get_collate_fn()
        train_loader=DataLoader(
            dataset=train_set,
            batch_size=4,
            shuffle=True,
            collate_fn=collate,
            drop_last=True)

        test_loader=DataLoader(
            dataset=test_set,
            batch_size=4,
            shuffle=True,
            collate_fn=collate,
            drop_last=True)

        val_loader=DataLoader(
            dataset=val_set,
            batch_size=4,
            shuffle=True,
            collate_fn=collate,
            drop_last=True)
        sample_text=format(val_data[1])
        torch.manual_seed(123)
        optimizer=get_optimizer(model=model)
        num_epochs = 4
        train_model(model=model,device=device,train_loader=train_loader,num_epoch=num_epochs,optimizer=optimizer,tokenizer=tokenizer,sample_text=sample_text)
        save_model(model=model,name="finetuned")
    else:
        print("Exiting !!!!!!!!!. No base model to load from")
        sys.exit()

if __name__ == "__main__":
    main()

