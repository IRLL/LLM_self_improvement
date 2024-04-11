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

from torchmetrics.text.rouge import ROUGEScore

from dataset_helpers import FinetuneDataset, NIevalDataset
from peft import PeftModel
from utils import log_method

@log_method
def finetune(model, tokenizer, result_save_path, answer_dataset, iter_version):
    torch.cuda.empty_cache()
    model.train()
    model.config.use_cache = False
    rouge = ROUGEScore()

    deepspeed_config_path = None
    rouge_result = []
    # Assuming your JSON data is in 'data.json', and located in the same directory as this script

    # Create dataset and dataloader
    finetune_dataset = FinetuneDataset(tokenizer, dataset=answer_dataset)
    nl_eval_dataset = NIevalDataset(tokenizer)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        np_array = torch.as_tensor(predictions)
        predictions = torch.argmax(np_array, dim=-1)
        labels = np.where(labels !=-100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Assuming labels are not already strings:
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        rouge_score = rouge(decoded_preds, decoded_labels)
        print(rouge_score)

        rouge_result.append({k: v.item() for k, v in rouge_score.items()})

        
        return {"rouge_score": rouge_score}

    target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj']#,'lm_head']
    lora_config = LoraConfig(r=16,
                target_modules = target_modules,
                lora_alpha=8,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM")

    #model.config.use_cache = True
    model.config.pretraining_tp = 1
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training settings
    training_params = TrainingArguments(
        output_dir=result_save_path,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        do_train=True,
        gradient_accumulation_steps=1,
        logging_steps=25,
        learning_rate=5e-5,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="none",
        evaluation_strategy="epoch",
        deepspeed=deepspeed_config_path,
        eval_accumulation_steps=2
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
    if iter_version ==0:
        metrics=trainer.evaluate()
        with open(os.path.join(result_save_path,"rouge_before_ft.json"),'w') as obj:
            obj.write(json.dumps(metrics))

    # Start training
    trainer.train()
    model_save_path = os.path.join(result_save_path, "model")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    model = model.merge_and_unload()

    with open(os.path.join(result_save_path,"rouge.json"),'w') as obj:
        obj.write(json.dumps(rouge_result))

    # metrics=trainer.evaluate()
    # print(metrics)
    del finetune_dataset
    del trainer
    del nl_eval_dataset
    model.config.use_cache = True
    return model, rouge_result