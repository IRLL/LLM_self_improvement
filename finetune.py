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
import sys

from dataset_helpers import FinetuneDataset
from peft import PeftModel
from utils import log_method,ClearCache,load_tokenizer,load_model_with_adapters

@log_method
def finetune():
    arguments = json.loads(sys.argv[1])

    iteration = int(arguments['cur_iteration'])
    adapters_path = arguments['adapters_path']
    result_save_path = arguments['result_save_path']
    model_path = arguments['model_path']
    feedback_dataset_path = arguments['feedback_dataset_path']
    model_adapter_save_path = arguments['model_adapter_save_path']

    with ClearCache():
        tokenizer = load_tokenizer(model_path)
        model = load_model_with_adapters(iteration, adapters_path, model_path)
        model.train()

        deepspeed_config_path = None
        # Assuming your JSON data is in 'data.json', and located in the same directory as this script

        # Create dataset and dataloader
        finetune_dataset = FinetuneDataset(tokenizer, filename=feedback_dataset_path)

        target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj']#,'lm_head']
        lora_config = LoraConfig(r=16,
                    target_modules = target_modules,
                    lora_alpha=8,
                    lora_dropout=0.05,
                    inference_mode=False,
                    bias="none",
                    task_type="CAUSAL_LM")

        #model.config.use_cache = False
        model.config.pretraining_tp = 1
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Training settings
        training_params = TrainingArguments(
            num_train_epochs=1,
            output_dir=result_save_path,
            do_train=True,
            do_eval=False,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            logging_steps=25,
            learning_rate=1e-3,
            weight_decay=0.001,
            per_device_eval_batch_size=2,
            fp16=True,
            bf16=False,
            max_grad_norm=1,
            max_steps=-1,
            group_by_length=True,
            lr_scheduler_type="constant",
            report_to="none",
            logging_dir=os.path.join(result_save_path,'loss_logs'),
            # evaluation_strategy="epoch",
            deepspeed=deepspeed_config_path,
            # eval_accumulation_steps=2,
        )
        # print(os.system("nvidia-smi"))
        # Initialize the Trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=finetune_dataset,
            peft_config=lora_config,
            dataset_text_field="text",
            max_seq_length=None,
            tokenizer=tokenizer,
            args=training_params,
            packing=False
            
        )

        # Start training
        trainer.train()
        model.save_pretrained(model_adapter_save_path)


        del finetune_dataset
        del trainer
        del training_params
finetune()