from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
import transformers
import torch
import os,tqdm,json
import datasets
from transformers.pipelines.pt_utils import KeyDataset
from torch.utils.data import Dataset

from utils import parse_arguments
from peft import PeftModel


# Load arguments.
args = parse_arguments()

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


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
                                             use_cache=False,device_map="auto")
if args.adapter_path:
    model = PeftModel.from_pretrained(model,model_id=args.adapter_path)
    model = model.merge_and_unload()

model.eval()
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    max_new_tokens=200,
    num_beams=5,
    num_return_sequences=1

)
with open(args.feedback_prompt_set_path) as obj:
    feedback_data = json.loads(obj.read())

texts = feedback_data["all_task_feedback_gen_prompt_data"]

result = []
idx = 0
for each in tqdm.tqdm(texts):
    res = "fake feedback"#pipeline(each)
    # output_text = res['generated_text'][len(texts[idx]):]
    # truncated_result = output_text.strip()
    # result.append(truncated_result)
    result.append(res)
    #print(result)

feedback_data["Feedback Label"]=[]

for i, text in enumerate(texts):
    # Write answer prediction to json file.
    feedback_data["Feedback Label"].append(result[i])
    #print(f"Input: {text}\nOutput: {result[i]['generated_text']}\n")

with open(args.feedback_dataset_path,"w") as obj:
    obj.write(json.dumps(feedback_data))