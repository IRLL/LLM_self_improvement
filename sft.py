

import json, os, torch

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

from dataset_helpers import SFTDataset, NIevalDataset
from peft import PeftModel
from utils import log_method
from utils import parse_arguments, load_model, load_tokenizer
from eval_boolq import eval_boolq
from squad_evaluation import eval_squad
from eval_math import eval_gsm8k


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
args = parse_arguments()


from transformers import TrainerCallback

# sbatch /home/qianxi/scratch/laffi/code/scripts/submit_sft_3b.sh ; sbatch /home/qianxi/scratch/laffi/code/scripts/submit_sft_7b.sh ; sbatch /home/qianxi/scratch/laffi/code/scripts/submit_sft_13b.sh

# sbatch /home/qianxi/scratch/laffi/code/scripts/submit_baseline_3b.sh ; sbatch /home/qianxi/scratch/laffi/code/scripts/submit_baseline_7b.sh ; sbatch /home/qianxi/scratch/laffi/code/scripts/submit_baseline_13b.sh
def evaluation(model,tokenizer, trainer, result_save_path):
    # metrics=trainer.evaluate()
    # with open(os.path.join(result_save_path,"rouge_before_ft.json"),'w') as obj:
    #     obj.write(json.dumps(metrics))
    # print("rouge:",metrics)
    math_result_path = os.path.join(result_save_path, "math.json")
    acc_before = eval_gsm8k(model, tokenizer, args.gsm8k_testset, gsm8k_eval_result_path=math_result_path)
    print("math:",acc_before)
    boolq_result = eval_boolq(model, tokenizer,
                                boolq_eval_path="/home/qianxi/scratch/laffi/datasets/boolq/eval_boolq.json",
                                boolq_eval_result_path=os.path.join(result_save_path, "boolq_eval_result.json"))
    print("boolq:",boolq_result)
    transformed_squad_eval_set_path = "/home/qianxi/scratch/laffi/datasets/squad2/truncated_processed_eval_dataset.json"
    original_squad_eval_set_path = "/home/qianxi/scratch/laffi/datasets/squad2/truncated_squal_eval.json"
    squad_response_gen_file = os.path.join(
        result_save_path, "squad_reponse_prediction.json")
    squad_eval_result_path = os.path.join(
        result_save_path, "squad_eval_result.json")

    squad_result = eval_squad(model,
                                tokenizer,
                                transformed_squad_eval_set_path,
                                original_squad_eval_set_path,
                                squad_response_gen_file,
                                squad_eval_result_path)
    
    print("squad_result:",squad_result)


class LossLoggingCallback(TrainerCallback):
    """A custom callback to log training losses."""

    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        # logs dictionary contains the training loss, validation loss, etc.
        if 'loss' in logs:  # Check if training loss is available
            self.losses.append(logs)
            # Save the updated losses list to a file
            with open(self.save_path, 'w') as f:
                json.dump(self.losses, f)

def finetune(model, tokenizer, result_save_path, sft_datset):
    torch.cuda.empty_cache()
    rouge = ROUGEScore()

    deepspeed_config_path = None
    rouge_result = []
    # Assuming your JSON data is in 'data.json', and located in the same directory as this script

    # Create dataset and dataloader
    finetune_dataset = SFTDataset(tokenizer, filename=sft_datset)
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
        del decoded_preds, decoded_labels,np_array,predictions
        
        return {"rouge_score": rouge_score}

    target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj']#,'lm_head']
    lora_config = LoraConfig(r=16,
                target_modules = target_modules,
                lora_alpha=8,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM")

    # Before ft:
     # Training settings
    training_params = TrainingArguments(
        output_dir=result_save_path,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        logging_steps=50,
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
        logging_dir=os.path.join(result_save_path,'loss_logs'),
        evaluation_strategy="epoch",
        deepspeed=deepspeed_config_path,
        eval_accumulation_steps=2
    )
    if args.baseline_only:
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
            compute_metrics=compute_metrics,
            callbacks=[LossLoggingCallback(os.path.join(result_save_path,'loss_logs.json'))]
            
        )
        evaluation(model,tokenizer, trainer, result_save_path)
        return None, None

    #model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

   
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
        compute_metrics=compute_metrics,
        callbacks=[LossLoggingCallback(os.path.join(result_save_path,'loss_logs.json'))]
        
    )

    torch.cuda.empty_cache()
    print("trainer before train",os.system("nvidia-smi"))
    # Start training
    trainer.train()
    model_save_path = os.path.join(result_save_path, "model")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    model = model.merge_and_unload()
    torch.cuda.empty_cache()
    # After finetuning
    evaluation(model,tokenizer,trainer, result_save_path)

    with open(os.path.join(result_save_path,"rouge.json"),'w') as obj:
        obj.write(json.dumps(rouge_result))

    # metrics=trainer.evaluate()
    # print(metrics)
    del finetune_dataset
    del trainer
    del nl_eval_dataset
    torch.cuda.empty_cache()

    return model, rouge_result

model = load_model(args.model_path, four_bit_quant=True, adapter_path=None)
tokenizer = load_tokenizer(args.model_path)
result_save_path = args.experiment_root_path





model, rouge_result = finetune(model, tokenizer, result_save_path, "/home/qianxi/scratch/laffi/datasets/natural_instruction_v1/natural_ins_train_50.json")
print(rouge_result)