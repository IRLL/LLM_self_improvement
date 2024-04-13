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
from peft import PeftModel

from utils import parse_arguments, load_model, load_tokenizer
base_model="/home/qianxi/scratch/laffi/models/7b"


adapter_folder = '/home/qianxi/scratch/laffi/code/test_load_model'
all_iter = 3
def load_model_with_adapters(all_iter, adapter_folder, base_model):
    model = load_model(base_model, four_bit_quant=True, adapter_path=None)

    adapter_name_list = []
    weight_list = []
    if all_iter >= 1:
        full_adapter_path = os.path.join(adapter_folder,f"model1")
        model = PeftModel.from_pretrained(model, full_adapter_path,adapter_name="model1")
        weight = 1
        adapter_name_list.append("model1")
        weight_list.append(weight)

        if all_iter > 1:
            for i in range(2, all_iter+1):
                full_adapter_path = os.path.join(adapter_folder,f"model{i}")
                weight = 1
                model.load_adapter(full_adapter_path,adapter_name=f"model{i}")
                adapter_name_list.append(f"model{i}")
                weight_list.append(weight)
                
            model.add_weighted_adapter(adapter_name_list, weight_list, "combined", combination_type="ties", density=0.2)

        # model = model.merge_and_unload()
    return model

