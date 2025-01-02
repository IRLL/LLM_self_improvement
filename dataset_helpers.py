"""
Author: Qianxi Li
Date: June 2, 2024
Description:
    This module defines PyTorch Dataset classes used for fine-tuning and 
    evaluating transformer-based models:
        1. FinetuneDataset: 
            - Loads data from a JSON file containing feedback inputs and labels 
              for supervised fine-tuning (SFT).
        2. NIevalDataset:
            - Loads Natural Instructions evaluation data from a JSON file.
        3. SFTDataset:
            - Loads an SFT dataset from a given JSON file.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader  # Using Dataset for custom PyTorch data handling.

class FinetuneDataset(Dataset):
    """
    FinetuneDataset:
        A PyTorch dataset class that reads a JSON file containing:
            - "all_feedback_input_list"
            - "Feedback Label"
        and tokenizes them for fine-tuning language models.
    """
    def __init__(self, tokenizer, filename="data.json", max_length=256):
        # Store the tokenizer instance for later use.
        self.tokenizer = tokenizer
        # Initialize lists to store model inputs and targets.
        self.inputs = []
        self.targets = []
        # Define a maximum sequence length for tokenization.
        self.max_length = max_length

        # Open the specified JSON file and load the contents.
        with open(filename, 'r') as file:
            data = json.load(file)  # Load JSON data into a dictionary.

            # Extract the lists of inputs and labels from the JSON keys.
            self.inputs = data["all_feedback_input_list"]
            self.targets = data["Feedback Label"]

    def __len__(self):
        # Return the total number of examples in the dataset.
        return len(self.inputs)

    def __getitem__(self, idx):
        # Tokenize the input at the given index.
        input_encodings = self.tokenizer(
            self.inputs[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )
        # Tokenize the target/label at the same index.
        target_encodings = self.tokenizer(
            self.targets[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )

        # Return a dictionary with tokenized input, attention mask, and labels.
        return {
            'input_ids': torch.tensor(input_encodings['input_ids']),
            'attention_mask': torch.tensor(input_encodings['attention_mask']),
            'labels': torch.tensor(target_encodings['input_ids'])
        }


class NIevalDataset(Dataset):
    """
    NIevalDataset:
        A PyTorch dataset class that reads a JSON file containing:
            - "input"
            - "label"
        and tokenizes them for evaluation on Natural Instructions tasks.
    """
    def __init__(self, tokenizer, filename, max_length=256):
        # Store the tokenizer instance for later use.
        self.tokenizer = tokenizer
        # Initialize lists to store model inputs and targets.
        self.inputs = []
        self.targets = []
        # Define a maximum sequence length for tokenization.
        self.max_length = max_length

        # Open the specified JSON file and load the contents.
        with open(filename, 'r') as file:
            data = json.load(file)  # Load JSON data into a dictionary.

            # Extract the lists of inputs and labels from the JSON keys.
            self.inputs = data["input"]
            self.targets = data["label"]

    def __len__(self):
        # Return the total number of examples in the dataset.
        return len(self.inputs)

    def __getitem__(self, idx):
        # Tokenize the input at the given index.
        input_encodings = self.tokenizer(
            self.inputs[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )
        # Tokenize the target/label at the same index.
        target_encodings = self.tokenizer(
            self.targets[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )

        # Return a dictionary with tokenized input, attention mask, and labels.
        return {
            'input_ids': torch.tensor(input_encodings['input_ids']),
            'attention_mask': torch.tensor(input_encodings['attention_mask']),
            'labels': torch.tensor(target_encodings['input_ids'])
        }


class SFTDataset(Dataset):
    """
    SFTDataset:
        A PyTorch dataset class that reads a JSON file containing:
            - "input"
            - "label"
        and tokenizes them for supervised fine-tuning.
    """
    def __init__(self, tokenizer, filename="/home/qianxi/scratch/laffi/datasets/natural_instruction_v1/natural_ins_train_50.json", max_length=256):
        # Store the tokenizer instance for later use.
        self.tokenizer = tokenizer
        # Initialize lists to store model inputs and targets.
        self.inputs = []
        self.targets = []
        # Define a maximum sequence length for tokenization.
        self.max_length = max_length

        # Open the specified JSON file and load the contents.
        with open(filename, 'r') as file:
            data = json.load(file)  # Load JSON data into a dictionary.

            # Extract the lists of inputs and labels from the JSON keys.
            self.inputs = data["input"]
            self.targets = data["label"]

    def __len__(self):
        # Return the total number of examples in the dataset.
        return len(self.inputs)

    def __getitem__(self, idx):
        # Tokenize the input at the given index.
        input_encodings = self.tokenizer(
            self.inputs[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )
        # Tokenize the target/label at the same index.
        target_encodings = self.tokenizer(
            self.targets[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )

        # Return a dictionary with tokenized input, attention mask, and labels.
        return {
            'input_ids': torch.tensor(input_encodings['input_ids']),
            'attention_mask': torch.tensor(input_encodings['attention_mask']),
            'labels': torch.tensor(target_encodings['input_ids'])
        }
