import argparse,json,torch,os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
from peft import PeftModel
import logging
import functools
from accelerate import infer_auto_device_map
import gc
import numpy as np
from transformers import BertTokenizer, BertModel
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class ClearCache:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        

    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()  # Optional: clears unused memory
            torch.cuda.synchronize() 
        

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

    parser.add_argument('--baseline_only', type=int, default=0,help='None')
    parser.add_argument("--enable_prompt_optimization", type=int, default=1, help="enable_prompt_optimization")
    parser.add_argument("--enable_initial_human_examples", type=int, default=1, help="enable_human_examples")
    parser.add_argument("--enable_mismatch_initial_human_examples", type=int, default=0, help="enable_mismatch_initial_human_examples")

    parser.add_argument("--model_path", type=str, default="/home/qianxi/scratch/laffi/models/7b", help="Path for the base dataset")

    
    parser.add_argument("--per_task_data_rows", type=int, default=10, help="How many training data rows to get from each task file")
    parser.add_argument("--num_return_seq", type=int, default=5, help="How many response to do major voting for the feedback.")
    parser.add_argument("--contamination", type=float, default=0.3, help="Outlier detection strength.")

    parser.add_argument("--eval_inference_batch_size", type=int, default=4, help="eval abtch size")


    parser.add_argument("--clusters", type=int, default=2, help="Number of cluster centers for this task.")

    parser.add_argument("--cur_iteration", type=int,default=0, help="Current Iteration #")
    parser.add_argument("--pos_example_amount", type=int, default=2, help="Number of positive examples for this task.")
    parser.add_argument("--neg_example_amount", type=int, default=0, help="Number of negative examples for this task.")
    parser.add_argument("--current_examples_path", type=str, default=None, help="Path for the base dataset")
    parser.add_argument("--adapter_path", type=str, default=None, help="Adapter path")

    # Natural Instruction related arguments
    parser.add_argument("--enable_natural_ins", type=int, default=1, help="If true, enable natural instruction evaluation")
    parser.add_argument("--na_ins_evalset_path", type=str, default="/home/qianxi/scratch/laffi/datasets/natural_instruction_v1/converted/natural_ins_eval_official_converted.json", help="Path for the eval dataset")

    # BoolQ related arguments
    parser.add_argument("--enable_boolq_eval", type=int, default=0, help="If true, enable boolq evaluation")
    parser.add_argument("--boolq_eval_path", type=str, default="/home/qianxi/scratch/laffi/datasets/boolq/eval_boolq.json", help="Boolq eval set path")
    parser.add_argument("--boolq_eval_result_path", type=str, default=None, help="Boolq eval result path")

    # Squad related arguments
    parser.add_argument("--enable_squad_eval", type=int, default=0, help="If true, enable squad evaluation")
    parser.add_argument("--transformed_squad_eval_set_path", type=str, default="/home/qianxi/scratch/laffi/datasets/squad2/processed_eval_dataset.json", help="Trans SQuAD eval set path")
    parser.add_argument("--original_squad_eval_set_path", type=str, default="/home/qianxi/scratch/laffi/datasets/squad2/squad_official_eval.json", help="Original SQuAD eval set path")
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

def load_bert():
    bert = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True).to("cuda:0")
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return bert, bert_tokenizer


def read_json(json_path):
    with open(json_path) as obj:
        data = json.loads(obj.read())

    return data

def write_json(json_path, content):
    with open(json_path,'w') as obj:
        obj.write(json.dumps(content,cls=NpEncoder))

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
def load_model_with_adapters(current_iter, adapter_folder, base_model):
    model = load_model(base_model, four_bit_quant=True)

    weight_list = []
    if current_iter >= 1:
        full_adapter_path = os.path.join(adapter_folder,"model1")
        model = PeftModel.from_pretrained(model, full_adapter_path, adapter_name="model1")
        final_name = "model1"
        if current_iter > 1:
            adapter_name_list = ["model1"]
            weight_list = [1.0]

            for i in range(2, current_iter+1):
                full_adapter_path = os.path.join(adapter_folder,f"model{i}")
                weight = 1
                model.load_adapter(full_adapter_path,adapter_name=f"model{i}")
                adapter_name_list.append(f"model{i}")
                weight_list.append(weight)
            
            final_name = "combined"
            model.add_weighted_adapter(adapter_name_list, weight_list, final_name, combination_type="ties", density=0.2)

        model = model.merge_and_unload()

    return model


def load_model(model_path, four_bit_quant):
    #device_map = infer_auto_device_map(my_model, max_memory={0: "10GiB", 1: "10GiB", "cpu": "30GiB"})
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
                                                device_map="auto")

    return model

def load_tokenizer(model_path):
    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer

def split_into_batches(lst, batch_size):
    """
    Splits a list of items into smaller batches of a specified size.
    
    Args:
    lst (list): The list of items to split.
    batch_size (int): The size of each batch.

    Returns:
    list of lists: A list where each item is a batch (list) of items.
    """
    # Handle boundary cases
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer")
    if not lst:
        return []  # Return an empty list if the input list is empty

    # Create batches
    batches = [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]
    return batches