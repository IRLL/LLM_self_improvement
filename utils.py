"""
Machine Learning Utilities Module

Author: Qianxi Li
Date: June 1, 2024
Description:
This module provides utility functions for machine learning operations including
model loading, tokenization, JSON handling, memory management, and metric calculations.
It also includes argument parsing and custom encoders for numpy types.
"""

import argparse
import json
import logging
import os
import functools
import gc
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from accelerate import infer_auto_device_map
from peft import PeftModel
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    BertTokenizer,
    BertModel
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NpEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for handling NumPy data types.
    
    This encoder converts NumPy types to their Python equivalents for JSON serialization.
    """
    
    def default(self, obj: Any) -> Any:
        """Convert NumPy types to Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class ClearCache:
    """
    Context manager for clearing GPU and system memory.
    
    Usage:
        with ClearCache():
            # Your memory-intensive operations here
    """
    
    def __enter__(self) -> None:
        """Clear memory on entering context."""
        gc.collect()
        torch.cuda.empty_cache()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Clear memory on exiting context and reset CUDA stats."""
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

def log_method(func: Any) -> Any:
    """
    Decorator to log the start and end of method execution.
    
    Args:
        func: Function to be decorated
        
    Returns:
        Wrapped function with logging
    """
    @functools.wraps(func)
    def wrapper_log_method(*args: Any, **kwargs: Any) -> Any:
        logger.info(f'Starting method {func.__name__}')
        result = func(*args, **kwargs)
        logger.info(f'Ending method {func.__name__}')
        return result
    return wrapper_log_method

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the application.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Fine-tuning LLM")
    
    # LaFFi related arguments
    parser.add_argument(
        "--feedback_dataset_path",
        type=str,
        help="Path for the feedback prediction dataset"
    )
    parser.add_argument(
        "--base_dataset_path",
        type=str,
        default="/home/qianxi/scratch/laffi/datasets/natural_instruction_v1/train",
        help="Path for the base dataset"
    )
    parser.add_argument(
        "--experiment_root_path",
        type=str,
        default="/home/qianxi/scratch/laffi/code/results/",
        help="Root directory for storing results"
    )
    parser.add_argument(
        "--baseline_only",
        type=int,
        default=0,
        help="Run baseline evaluation only"
    )
    parser.add_argument(
        "--enable_prompt_optimization",
        type=int,
        default=1,
        help="Enable prompt optimization"
    )
    parser.add_argument(
        "--enable_initial_human_examples",
        type=int,
        default=1,
        help="Enable human examples"
    )
    parser.add_argument(
        "--enable_mismatch_initial_human_examples",
        type=int,
        default=0,
        help="Enable mismatched initial human examples"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/qianxi/scratch/laffi/models/7b",
        help="Path to the model"
    )
    parser.add_argument(
        "--per_task_data_rows",
        type=int,
        default=10,
        help="Number of training data rows per task file"
    )
    parser.add_argument(
        "--num_return_seq",
        type=int,
        default=5,
        help="Number of responses for major voting"
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.3,
        help="Outlier detection strength"
    )
    parser.add_argument(
        "--eval_inference_batch_size",
        type=int,
        default=4,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=2,
        help="Number of cluster centers for this task"
    )
    parser.add_argument(
        "--cur_iteration",
        type=int,
        default=0,
        help="Current iteration number"
    )
    parser.add_argument(
        "--pos_example_amount",
        type=int,
        default=2,
        help="Number of positive examples"
    )
    parser.add_argument(
        "--neg_example_amount",
        type=int,
        default=0,
        help="Number of negative examples"
    )
    parser.add_argument(
        "--current_examples_path",
        type=str,
        default=None,
        help="Path to current examples"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to adapter"
    )
    
    # Natural Instruction arguments
    parser.add_argument(
        "--enable_natural_ins",
        type=int,
        default=1,
        help="Enable natural instruction evaluation"
    )
    parser.add_argument(
        "--na_ins_evalset_path",
        type=str,
        default="/home/qianxi/scratch/laffi/datasets/natural_instruction_v1/converted/natural_ins_eval_official_converted.json",
        help="Path to evaluation dataset"
    )
    
    # BoolQ related arguments
    parser.add_argument(
        "--enable_boolq_eval",
        type=int,
        default=0,
        help="Enable BoolQ evaluation"
    )
    parser.add_argument(
        "--boolq_eval_path",
        type=str,
        default="/home/qianxi/scratch/laffi/datasets/boolq/eval_boolq.json",
        help="BoolQ evaluation set path"
    )
    parser.add_argument(
        "--boolq_eval_result_path",
        type=str,
        default=None,
        help="BoolQ evaluation result path"
    )
    
    # Squad related arguments
    parser.add_argument(
        "--enable_squad_eval",
        type=int,
        default=0,
        help="Enable SQUAD evaluation"
    )
    parser.add_argument(
        "--transformed_squad_eval_set_path",
        type=str,
        default="/home/qianxi/scratch/laffi/datasets/squad2/processed_eval_dataset.json",
        help="Transformed SQUAD evaluation set path"
    )
    parser.add_argument(
        "--original_squad_eval_set_path",
        type=str,
        default="/home/qianxi/scratch/laffi/datasets/squad2/squad_official_eval.json",
        help="Original SQUAD evaluation set path"
    )
    parser.add_argument(
        "--squad_response_gen_file",
        type=str,
        default=None,
        help="SQUAD response generation file"
    )
    parser.add_argument(
        "--squad_eval_result_path",
        type=str,
        default=None,
        help="SQUAD evaluation result path"
    )
    
    # GSM8K related arguments
    parser.add_argument(
        "--enable_gsm8k_eval",
        type=int,
        default=0,
        help="Enable GSM8K evaluation"
    )
    parser.add_argument(
        "--gsm8k_testset",
        type=str,
        default="/home/qianxi/scratch/laffi/datasets/GSM8K/grade_school_math/data/test.json",
        help="GSM8K test set path"
    )
    
    return parser.parse_args()

def load_bert() -> Tuple[BertModel, BertTokenizer]:
    """
    Load BERT model and tokenizer.
    
    Returns:
        Tuple of (BERT model, BERT tokenizer)
    """
    bert = BertModel.from_pretrained(
        'bert-base-uncased',
        output_hidden_states=True
    ).to("cuda:0")
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return bert, bert_tokenizer

def read_json(json_path: str) -> Dict:
    """
    Read and parse JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Parsed JSON content
    """
    with open(json_path) as obj:
        data = json.loads(obj.read())
    return data

def write_json(json_path: str, content: Any) -> None:
    """
    Write content to JSON file using custom NumPy encoder.
    
    Args:
        json_path: Path to save JSON file
        content: Content to save
    """
    with open(json_path, 'w') as obj:
        json.dump(content, obj, cls=NpEncoder)

def calculate_classification_metrics(
    predictions: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Calculate classification metrics for boolean question answering.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        
    Returns:
        Dictionary of metrics (precision, recall, F1, accuracy)
    """
    return {
        "boolq_precision": precision_score(labels, predictions),
        "boolq_recall": recall_score(labels, predictions),
        "boolq_f1_score": f1_score(labels, predictions),
        "boolq_accuracy": accuracy_score(labels, predictions)
    }

@log_method
def load_model_with_adapters(
    current_iter: int,
    adapter_folder: str,
    base_model: str
) -> Any:
    """
    Load model with multiple adapters and combine them.
    
    Args:
        current_iter: Current iteration number
        adapter_folder: Path to adapter folder
        base_model: Path to base model
        
    Returns:
        Model with combined adapters
    """
    model = load_model(base_model, four_bit_quant=True)

    if current_iter >= 1:
        full_adapter_path = os.path.join(adapter_folder, "model1")
        model = PeftModel.from_pretrained(model, full_adapter_path, adapter_name="model1")
        final_name = "model1"

        if current_iter > 1:
            adapter_name_list = ["model1"]
            weight_list = [1.0]

            for i in range(2, current_iter + 1):
                full_adapter_path = os.path.join(adapter_folder, f"model{i}")
                model.load_adapter(full_adapter_path, adapter_name=f"model{i}")
                adapter_name_list.append(f"model{i}")
                weight_list.append(1.0)

            final_name = "combined"
            model.add_weighted_adapter(
                adapter_name_list,
                weight_list,
                final_name,
                combination_type="ties",
                density=0.2
            )

        model = model.merge_and_unload()

    return model

def load_model(
    model_path: str,
    four_bit_quant: bool
) -> AutoModelForCausalLM:
    """
    Load pre-trained model with optional quantization.
    
    Args:
        model_path: Path to model
        four_bit_quant: Whether to use 4-bit quantization
        
    Returns:
        Loaded model
    """
    quant_config = None
    if four_bit_quant:
        compute_dtype = getattr(torch, "float16")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    return model

def load_tokenizer(model_path: str) -> AutoTokenizer:
    """
    Load tokenizer for the model.
    
    Args:
        model_path: Path to model
        
    Returns:
        Loaded tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer

def split_into_batches(lst: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split a list into batches of specified size.
    
    Args:
        lst: List to split
        batch_size: Size of each batch
        
    Returns:
        List of batches
        
    Raises:
        ValueError: If batch_size is not positive
    """
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer")
    if not lst:
        return []

    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]