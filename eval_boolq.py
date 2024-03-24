from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
import transformers
import torch
import os,tqdm,json
# import datasets
# from transformers.pipelines.pt_utils import KeyDataset
# from torch.utils.data import Dataset
# import accelerate

from utils import parse_arguments,calculate_classification_metrics
from peft import PeftModel


# Load arguments.
args = parse_arguments()
assert args.boolq_eval_result_path,"Must exist"
assert args.model_path,"Must exist"
assert args.boolq_eval_path,"Must exist"


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
    max_new_tokens=20


)
with open(args.boolq_eval_path) as obj:
    boolq_data = json.loads(obj.read())

#print(len(boolq_data))
boolq_data =boolq_data[:10]

prompt = """Write a response that appropriately completes answer the question, follow the examples. Your answer should be "True" or "False".

### Example 1:
Passage: 
The Vampire Diaries, an American supernatural drama, was renewed for an eighth season by The CW on March 11, 2016. On July 23, 2016, the CW announced that the upcoming season would be the series' last and would consist of 16 episodes. The season premiered on October 21, 2016 and concluded on March 10, 2017.

Question:
will there be a season 8 of vampire diaries?

Answer:
True

### Example 2:
Passage: 
This is the list of U.S. states that have participated in the Little League World Series. As of the 2018 LLWS, eight states had never reached the LLWS: Alaska, Colorado, Kansas, North Dakota, Utah, Vermont, Wisconsin, and Wyoming; additionally, the District of Columbia has never reached the LLWS.

Question:
has wisconsin ever been in the little league world series?

Answer:
False

### Task:
Passage:
{passage}

Question:
{question}

Answer:"""

predictions = []
labels = []
for idx, item in tqdm.tqdm(enumerate(boolq_data)): 
    full_prompt = prompt.format(question=item['question'], passage=item['passage'])
    result = pipeline(full_prompt)
    #print("full_prompt",full_prompt)
    truncated = result[0]['generated_text'][len(full_prompt):].strip()
    #print(truncated)
    if "false" in truncated or "False" in truncated:
        predictions.append(0)
    else:
        predictions.append(1)
    labels.append(item["answer"])
# print(predictions, labels)
metrics = calculate_classification_metrics(predictions, labels)

with open(args.boolq_eval_result_path,'w') as obj:
    obj.write(json.dumps(metrics))
