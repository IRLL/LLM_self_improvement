import os
import json
import torch
from torch.utils.data import Dataset, DataLoader


class FinetuneDataset(Dataset):
    def __init__(self, tokenizer, filename="data.json", max_length=256):
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.max_length = max_length
        with open(filename, 'r') as file:
            data = json.load(file)
            
            self.inputs = data["all_feedback_input_list"]
            self.targets= data["Feedback Label"]


                

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_encodings = self.tokenizer(self.inputs[idx], truncation=True, padding='max_length', max_length=self.max_length)
        target_encodings = self.tokenizer(self.targets[idx], truncation=True, padding='max_length', max_length=self.max_length)

        return {
            'input_ids': torch.tensor(input_encodings['input_ids']),
            'attention_mask': torch.tensor(input_encodings['attention_mask']),
            'labels': torch.tensor(target_encodings['input_ids'])
        }


class NIevalDataset(Dataset):
    def __init__(self, tokenizer, filename, max_length=256):
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.max_length = max_length
        with open(filename, 'r') as file:
            data = json.load(file)
        
            self.inputs = data["input"]
            self.targets= data["label"]


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_encodings = self.tokenizer(self.inputs[idx], truncation=True, padding='max_length', max_length=self.max_length)
        target_encodings = self.tokenizer(self.targets[idx], truncation=True, padding='max_length', max_length=self.max_length)

        return {
            'input_ids': torch.tensor(input_encodings['input_ids']),
            'attention_mask': torch.tensor(input_encodings['attention_mask']),
            'labels': torch.tensor(target_encodings['input_ids'])
        }


class SFTDataset(Dataset):
    def __init__(self, tokenizer, filename="/home/qianxi/scratch/laffi/datasets/natural_instruction_v1/natural_ins_train_50.json", max_length=256):
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.max_length = max_length
        with open(filename, 'r') as file:
            data = json.load(file)
        
            self.inputs = data["input"]
            self.targets= data["label"]


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_encodings = self.tokenizer(self.inputs[idx], truncation=True, padding='max_length', max_length=self.max_length)
        target_encodings = self.tokenizer(self.targets[idx], truncation=True, padding='max_length', max_length=self.max_length)

        return {
            'input_ids': torch.tensor(input_encodings['input_ids']),
            'attention_mask': torch.tensor(input_encodings['attention_mask']),
            'labels': torch.tensor(target_encodings['input_ids'])
        }