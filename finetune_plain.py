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

from datasets import load_dataset
# from accelerate import PartialState



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
model_name = "/home/qianxi/scratch/laffi/models/7b"  # Replace with your LLaMA 2 model name

tokenizer = AutoTokenizer.from_pretrained(model_name,device_map="auto")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
# Create dataset and dataloader
# dataset = CustomDataset(tokenizer, filename="/home/qianxi/scratch/laffi/datasets/test.json")
dataset = load_dataset("mlabonne/guanaco-llama2-1k", split="train")
# def tokenization(example):
#     return tokenizer(example["text"])

# dataset = dataset.map(tokenization, batched=True)
print("after load dataset")
print(os.system("nvidia-smi"))
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
print("before load model")
print(os.system("nvidia-smi"))
# Assuming we're proceeding normally without these adjustments for demonstration:
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             quantization_config=quant_config,
                                             low_cpu_mem_usage=True,
                                             device_map="auto")
print(model)
print("before lora")
print(os.system("nvidia-smi"))
model.config.pretraining_tp = 1
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ds_config = DeepSpeedConfig(deepspeed_config_path)


# Training settings
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="none"

)
print("before sft trainer")
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
print("before trainer")
print(os.system("nvidia-smi"))
# Start training
trainer.train()
