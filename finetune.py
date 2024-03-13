import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,

    default_data_collator
)
from trl import SFTTrainer
from peft import LoraConfig,get_peft_model
# from datasets import load_metric
import deepspeed
from datasets import load_dataset
from accelerate import PartialState
import argparse
from datetime import datetime
from deepspeed.runtime.config import DeepSpeedConfig



# Format the date and time as a string
date_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tuning a Language Model")
    parser.add_argument("--enable_ds", type=int, default=0, help="Whether to use deepspeed for finetuning.")
    parser.add_argument("--ds_config_path", type=str, default="/home/qianxi/scratch/laffi/code/ds_config.json", help="ds config path")
    parser.add_argument("--parent_root", type=str, default="/home/qianxi/scratch/laffi", help="Root directory for the project")
    parser.add_argument("--model_name", type=str, default="7b", help="Model name or path")
    # parser.add_argument("--dataset_name", type=str, default="mlabonne/guanaco-llam   a2-1k", help="Dataset name or path")
    # parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    # parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    # parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    # parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay")
    # parser.add_argument("--max_grad_norm", type=float, default=0.3, help="Max grad norm")
    # parser.add_argument("--lora_r", type=int, default=16, help="Rank for LoRA")
    # parser.add_argument("--lora_alpha", type=int, default=8, help="Scale factor for LoRA")
    # parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout rate for LoRA")
    # parser.add_argument("--use_fp16", action='store_true', help="Use mixed precision training")
    # Add other arguments as needed
    return parser.parse_args()

args = parse_arguments()

parent_root = "/home/qianxi/scratch/laffi"

model_path = os.path.join(parent_root,f"models/{args.model_name}")
result_path = os.path.join(parent_root,f"results/{args.model_name}-{date_time_str}")

if args.enable_ds:
    deepspeed_config_path = args.ds_config_path
    ds_config=DeepSpeedConfig(json_file=deepspeed_config_path)
    device_map={'':PartialState().process_index}

else: 
    deepspeed_config_path = None
    ds_config = None
    device_map="auto"
# Assuming your JSON data is in 'data.json', and located in the same directory as this script
class CustomDataset(Dataset):
    def __init__(self, tokenizer, filename="data.json", max_length=256):
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.max_length = max_length
        with open(filename, 'r') as file:
            data = json.load(file)
            for key,value in data.items():
                self.inputs.append(key)
                self.targets.append(value)


                

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

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
# Create dataset and dataloader
# dataset = CustomDataset(tokenizer, filename="/home/qianxi/scratch/laffi/datasets/test.json")
dataset = load_dataset("mlabonne/guanaco-llama2-1k", split="train")
# def tokenization(example):
#     return tokenizer(example["text"])

# dataset = dataset.map(tokenization, batched=True)

#dataloader = DataLoader(dataset, batch_size=4)  # Adjust batch size as needed
target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj']#,'lm_head']
lora_config = LoraConfig(r=16,
            target_modules = target_modules,
            lora_alpha=8,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM")



# Initialize model with LORA or QLORA modifications - Note: This part is more complex and may require 
# custom adjustments based on the libraries you use and their integration with Transformers.
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)
# Assuming we're proceeding normally without these adjustments for demonstration:
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             quantization_config=quant_config,
                                             low_cpu_mem_usage=True,
                                             use_cache=False,
                                             device_map=device_map)


model.config.use_cache = False
model.config.pretraining_tp = 1
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 


# Training settings
training_params = TrainingArguments(
    output_dir=result_path,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="none",
    deepspeed=deepspeed_config_path
)
print(os.system("nvidia-smi"))
# Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

# Start training
trainer.train()
