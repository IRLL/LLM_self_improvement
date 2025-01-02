"""
Supervised Fine-Tuning (SFT) Module

Author: Qianxi Li
Date: June 1, 2024
Description:
This module implements supervised fine-tuning for language models with adapter support.
It includes functionality for training, evaluation, and metric tracking across multiple tasks
including mathematical reasoning, boolean question answering, and squad-based tasks.
"""

import json
import os
import logging
from typing import Tuple, Optional, Any, Dict

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    default_data_collator,
    TrainerCallback
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, PeftModel
from torchmetrics.text.rouge import ROUGEScore

from dataset_helpers import SFTDataset, NIevalDataset
from utils import log_method, parse_arguments, load_model, load_tokenizer
from eval_boolq import eval_boolq
from squad_evaluation import eval_squad
from eval_math import eval_gsm8k

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enable CUDNN optimizations
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Parse command line arguments
args = parse_arguments()

def evaluation(
    model: Any,
    tokenizer: Any,
    trainer: Any,
    result_save_path: str
) -> Dict[str, float]:
    """
    Evaluate model performance across multiple tasks.
    
    Args:
        model: The trained model
        tokenizer: Tokenizer for the model
        trainer: Training handler
        result_save_path: Path to save evaluation results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Evaluate mathematical reasoning
    math_result_path = os.path.join(result_save_path, "math.json")
    math_acc = eval_gsm8k(
        model,
        tokenizer,
        args.gsm8k_testset,
        gsm8k_eval_result_path=math_result_path
    )
    logger.info(f"Math evaluation accuracy: {math_acc}")

    # Evaluate boolean question answering
    boolq_result = eval_boolq(
        model,
        tokenizer,
        boolq_eval_path="/home/qianxi/scratch/laffi/datasets/boolq/eval_boolq.json",
        boolq_eval_result_path=os.path.join(result_save_path, "boolq_eval_result.json")
    )
    logger.info(f"BoolQ evaluation results: {boolq_result}")

    # Evaluate SQuAD performance
    squad_result = eval_squad(
        model,
        tokenizer,
        args.transformed_squad_eval_set_path,
        args.original_squad_eval_set_path,
        os.path.join(result_save_path, "squad_reponse_prediction.json"),
        os.path.join(result_save_path, "squad_eval_result.json")
    )
    logger.info(f"SQUAD evaluation results: {squad_result}")

    return {
        "math_accuracy": math_acc,
        "boolq_results": boolq_result,
        "squad_results": squad_result
    }

class LossLoggingCallback(TrainerCallback):
    """
    Custom callback for logging training losses during model training.
    
    Attributes:
        save_path: Path to save the loss logs
        losses: List to store loss values
    """

    def __init__(self, save_path: str):
        """Initialize the callback with save path."""
        super().__init__()
        self.save_path = save_path
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training losses and save to file."""
        if logs and 'loss' in logs:
            self.losses.append(logs)
            with open(self.save_path, 'w') as f:
                json.dump(self.losses, f)

def finetune(
    model: Any,
    tokenizer: Any,
    result_save_path: str,
    sft_dataset: str
) -> Tuple[Optional[Any], Optional[list]]:
    """
    Fine-tune the model using supervised learning.
    
    Args:
        model: Base model to fine-tune
        tokenizer: Tokenizer for the model
        result_save_path: Path to save results
        sft_dataset: Path to the SFT dataset
        
    Returns:
        Tuple of (fine-tuned model, ROUGE scores)
    """
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Initialize metrics
    rouge = ROUGEScore()
    rouge_result = []

    # Prepare datasets
    finetune_dataset = SFTDataset(tokenizer, filename=sft_dataset)
    nl_eval_dataset = NIevalDataset(tokenizer)

    def compute_metrics(eval_pred):
        """Compute ROUGE metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = torch.argmax(torch.as_tensor(predictions), dim=-1)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        rouge_score = rouge(decoded_preds, decoded_labels)
        rouge_result.append({k: v.item() for k, v in rouge_score.items()})
        
        return {"rouge_score": rouge_score}

    # Configure LoRA parameters
    target_modules = [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'down_proj', 'up_proj'
    ]
    lora_config = LoraConfig(
        r=16,
        target_modules=target_modules,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Configure training parameters
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
        logging_dir=os.path.join(result_save_path, 'loss_logs'),
        evaluation_strategy="epoch",
        eval_accumulation_steps=2
    )

    # Handle baseline-only case
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
            callbacks=[LossLoggingCallback(
                os.path.join(result_save_path, 'loss_logs.json')
            )]
        )
        evaluation(model, tokenizer, trainer, result_save_path)
        return None, None

    # Prepare model for training
    model.config.pretraining_tp = 1
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Initialize trainer
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
        callbacks=[LossLoggingCallback(
            os.path.join(result_save_path, 'loss_logs.json')
        )]
    )

    # Train model
    logger.info("Starting training process")
    trainer.train()
    
    # Save model and results
    model_save_path = os.path.join(result_save_path, "model")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # Merge and evaluate
    model = model.merge_and_unload()
    evaluation(model, tokenizer, trainer, result_save_path)

    # Save ROUGE scores
    with open(os.path.join(result_save_path, "rouge.json"), 'w') as obj:
        json.dump(rouge_result, obj)

    # Clean up
    del finetune_dataset
    del trainer
    del nl_eval_dataset
    torch.cuda.empty_cache()

    return model, rouge_result

def main():
    """Main execution function."""
    # Initialize model and tokenizer
    model = load_model(args.model_path, four_bit_quant=True, adapter_path=None)
    tokenizer = load_tokenizer(args.model_path)
    
    # Run fine-tuning
    model, rouge_result = finetune(
        model,
        tokenizer,
        args.experiment_root_path,
        "/home/qianxi/scratch/laffi/datasets/natural_instruction_v1/natural_ins_train_50.json"
    )
    
    logger.info(f"Final ROUGE results: {rouge_result}")

if __name__ == "__main__":
    main()