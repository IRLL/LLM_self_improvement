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

    parser.add_argument("--model_path", type=str, default="/home/qianxi/scratch/laffi/models/7b", help="Path for the base dataset")

    
    parser.add_argument("--iteration_amount", type=int,default=3, help="Iteration #")
    parser.add_argument("--pos_example_amount", type=int, default=2, help="Number of positive examples for this task.")
    parser.add_argument("--neg_example_amount", type=int, default=0, help="Number of negative examples for this task.")
    parser.add_argument("--current_examples_path", type=str, default=None, help="Path for the base dataset")
    parser.add_argument("--adapter_path", type=str, default=None, help="Adapter path")

    # BoolQ related arguments
    parser.add_argument("--boolq_eval_path", type=str, default=None, help="Boolq eval set path")
    parser.add_argument("--boolq_eval_result_path", type=str, default=None, help="Boolq eval result path")

    # Squad related arguments
    parser.add_argument("--transformed_squad_eval_set_path", type=str, default=None, help="Trans SQuAD eval set path")
    parser.add_argument("--original_squad_eval_set_path", type=str, default=None, help="Original SQuAD eval set path")
    parser.add_argument("--squad_response_gen_file", type=str, default=None, help="squad_response_gen_file")
    parser.add_argument("--squad_eval_result_path", type=str, default=None, help="squad_eval_result_path")

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
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy
    }

def replace_cur_examples(feedback_dataset_path, new_example_indices_dict):
    with open(feedback_dataset_path) as obj:
        example_json = json.loads(obj.read())

    for key in example_json.keys():
        pass

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


    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                quantization_config=quant_config,
                                                low_cpu_mem_usage=True,
                                                use_cache=False,
                                                device_map="auto")

    if adapter_path:
        model = PeftModel.from_pretrained(model,model_id=adapter_path)
        model = model.merge_and_unload() 

    return model

def load_tokenizer(model_path):
    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer