import os
import json
import torch
import numpy as np
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
from torchmetrics.text.rouge import ROUGEScore

from dataset_helpers import FinetuneDataset, NIevalDataset
from peft import PeftModel




# Format the date and time as a string
date_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tuning a Language Model")
    parser.add_argument("--enable_ds", type=int, default=0, help="Whether to use deepspeed for finetuning.")
    parser.add_argument("--ds_config_path", type=str, default="/home/qianxi/scratch/laffi/code/ds_config.json", help="ds config path")
    parser.add_argument("--parent_root", type=str, default="/home/qianxi/scratch/laffi", help="Root directory for the project")
    parser.add_argument("--model_name", type=str, default="7b", help="Model name or path")
    parser.add_argument("--dataset_path", type=str, default="mlabonne/guanaco-llam   a2-1k", help="Dataset path")
    parser.add_argument("--model_save_path", type=str, default="/result", help="Result path")
    parser.add_argument("--adapter_path", type=str, default=None, help="Adapter path")
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
rouge = ROUGEScore()
parent_root = args.parent_root

model_path = os.path.join(parent_root,f"models/{args.model_name}")
result_path = args.model_save_path

if args.enable_ds:
    deepspeed_config_path = args.ds_config_path
    ds_config=DeepSpeedConfig(json_file=deepspeed_config_path)
    device_map={'':PartialState().process_index}

else: 
    deepspeed_config_path = None
    ds_config = None
    device_map="auto"
# Assuming your JSON data is in 'data.json', and located in the same directory as this script

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
# Create dataset and dataloader
finetune_dataset = FinetuneDataset(tokenizer, filename=args.dataset_path)
nl_eval_dataset = NIevalDataset(tokenizer)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    np_array = torch.as_tensor(predictions)
    predictions = torch.argmax(np_array, dim=-1)
    labels = np.where(labels !=-100, labels, tokenizer.pad_token_id)
    # print(predictions.shape)
    # print(labels.shape)
    # print(labels[0])
    # # print(predictions[0])
    # assert 1==2
    # print(predictions.shape, labels.shape)
    torch.save(predictions, 'predictions.pt')
    torch.save(labels, 'labels.pt')
    assert 1==2
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Assuming labels are not already strings:
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    rouge_score = rouge(decoded_preds, decoded_labels)
    return {"rouge_score": rouge_score}

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
if args.adapter_path:
    model = PeftModel.from_pretrained(model,model_id=args.adapter_path)
    model = model.merge_and_unload()

model.config.use_cache = False
model.config.pretraining_tp = 1
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 


# Training settings
training_params = TrainingArguments(
    output_dir=result_path,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    save_steps=25,
    eval_steps=3,
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
    evaluation_strategy="steps",#epoch
    deepspeed=deepspeed_config_path,
    eval_accumulation_steps=4
)
# print(os.system("nvidia-smi"))
# Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=finetune_dataset,
    eval_dataset=nl_eval_dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
    compute_metrics=compute_metrics
    
)

# Start training
trainer.train()
model.save_pretrained(result_path)
# metrics=trainer.evaluate()
# print(metrics)