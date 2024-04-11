import argparse,json,torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
from peft import PeftModel
import logging
import functools


def log_method(func):
    """Decorator to log the start and end of a method."""
    @functools.wraps(func)
    def wrapper_log_method(*args, **kwargs):
        logging.info(f'Starting method {func.__name__}')
        result = func(*args, **kwargs)
        logging.info(f'Ending method {func.__name__}')
        return result
    return wrapper_log_method

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tuning LLM")
    # LaFFi related logic
    parser.add_argument("--feedback_dataset_path", type=str, help="Path for the feedback prediction dataset")
    parser.add_argument("--base_dataset_path", default="/home/qianxi/scratch/laffi/datasets/natural_instruction_v1/train", type=str, help="Path for the base dataset")
    parser.add_argument('--experiment_root_path', type=str, default="/home/qianxi/scratch/laffi/code/results/",help='Root directory for storing results.')
    parser.add_argument('--experiment_name', type=str, default="official",help='This experiment name')

    parser.add_argument("--model_path", type=str, default="/home/qianxi/scratch/laffi/models/7b", help="Path for the base dataset")
    parser.add_argument("--wandb_enabled", type=int, default=0, help="If 0, disable wandb")
    
    parser.add_argument("--per_task_data_rows", type=int, default=10, help="How many training data rows to get from each task file")
    parser.add_argument("--num_return_seq", type=int, default=8, help="How many response to do major voting for the feedback.")
    parser.add_argument("--contamination", type=float, default=0.3, help="Outlier detection strength.")

    parser.add_argument("--clusters", type=int, default=2, help="Number of cluster centers for this task.")
    parser.add_argument("--iteration_amount", type=int,default=2, help="Iteration #")
    parser.add_argument("--pos_example_amount", type=int, default=2, help="Number of positive examples for this task.")
    parser.add_argument("--neg_example_amount", type=int, default=0, help="Number of negative examples for this task.")
    parser.add_argument("--current_examples_path", type=str, default=None, help="Path for the base dataset")
    parser.add_argument("--adapter_path", type=str, default=None, help="Adapter path")

    # BoolQ related arguments
    parser.add_argument("--enable_boolq_eval", type=int, default=0, help="If true, enable boolq evaluation")
    parser.add_argument("--boolq_eval_path", type=str, default=None, help="Boolq eval set path")
    parser.add_argument("--boolq_eval_result_path", type=str, default=None, help="Boolq eval result path")

    # Squad related arguments
    parser.add_argument("--enable_squad_eval", type=int, default=0, help="If true, enable squad evaluation")
    parser.add_argument("--transformed_squad_eval_set_path", type=str, default=None, help="Trans SQuAD eval set path")
    parser.add_argument("--original_squad_eval_set_path", type=str, default=None, help="Original SQuAD eval set path")
    parser.add_argument("--squad_response_gen_file", type=str, default=None, help="squad_response_gen_file")
    parser.add_argument("--squad_eval_result_path", type=str, default=None, help="squad_eval_result_path")

    #gsm8k related arguments:
    parser.add_argument("--enable_gsm8k_eval", type=int, default=0, help="If true, enable gsm8k evaluation")
    parser.add_argument("--gsm8k_testset", type=str, default="/home/qianxi/scratch/laffi/datasets/GSM8K/grade_school_math/data/test.json", help="gsm8k_testset")


    # Finetuning related arguments.

    # parser.add_argument("--enable_ds", type=int, default=0, help="Whether to use deepspeed for finetuning.")
    # parser.add_argument("--ds_config_path", type=str, default="/home/qianxi/scratch/laffi/code/ds_config.json", help="ds config path")
    # parser.add_argument("--parent_root", type=str, default="/home/qianxi/scratch/laffi", help="Root directory for the project")


    return parser.parse_args()

def calculate_classification_metrics(predictions, labels):
    # Calculate precision
    precision = precision_score(labels, predictions)
    # Calculate recall
    recall = recall_score(labels, predictions)
    # Calculate F1 score
    f1 = f1_score(labels, predictions)
    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)
    
    return {
        "boolq_precision": precision,
        "boolq_recall": recall,
        "boolq_f1_score": f1,
        "boolq_accuracy": accuracy
    }


@log_method
def load_model(model_path, four_bit_quant, adapter_path=None):
    quant_config=None
    if four_bit_quant:
        # Quantization settings.
        compute_dtype = getattr(torch, "float16")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )
    n_gpus = torch.cuda.device_count()
    max_memory = {}
    if n_gpus >= 2:

        max_memory[0] = "5GIB"
        max_memory[1] = "16GIB"
    else:
        max_memory[0] = "16GIB"

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                quantization_config=quant_config,
                                                #load_in_8bit=True,
                                                low_cpu_mem_usage=True,
                                                max_memory=max_memory,
                                                use_cache=True,
                                                device_map="sequential")


    if adapter_path:
        model = PeftModel.from_pretrained(model,model_id=adapter_path)
        model = model.merge_and_unload() 

    return model

def load_tokenizer(model_path):
    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer