from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
import transformers
import torch
import os,tqdm,json
import datasets
from transformers.pipelines.pt_utils import KeyDataset
from torch.utils.data import Dataset

from utils import parse_arguments

# Load arguments.
args = parse_arguments()

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Quantization settings.
compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                             quantization_config=quant_config,
                                             low_cpu_mem_usage=True,
                                             use_cache=False,
                                             device_map="auto")

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    max_new_tokens=200, 

)
with open(args.answer_prompt_set_path) as obj:
    answer_data = json.loads(obj.read())

texts = []
index_dict = []
for key, value in answer_data.items():
    texts += value['Answer Prediction Prompt Dataset']
    for each_instance_idx in range(len(value['Instances'])):
        index_dict.append((key, each_instance_idx))

    assert len(value['Answer Prediction Prompt Dataset']) == len(value['Instances'])
print("texts",len(texts))

result = []
idx = 0
for each in tqdm.tqdm(texts):
    res = "ahhahahha"#pipeline(each)
    # output_text = res['generated_text'][len(texts[idx]):]
    # truncated_result = output_text.strip()
    # result.append(truncated_result)
    # print(result)
    result.append(res)

for i, text in enumerate(texts):
    task, index = index_dict[i]
    # Write answer prediction to json file.
    answer_data[task]["Instances"][index]['answer_prediction'] = result[i]
    #print(f"Input: {text}\nOutput: {result[i]['generated_text']}\n")

with open(args.answer_dataset_path,"w") as obj:
    obj.write(json.dumps(answer_data))

del model
del pipeline
del tokenizer
del result