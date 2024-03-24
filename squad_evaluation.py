from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
import transformers
import torch
import os,json
from tqdm import tqdm
from datasets import Dataset

from transformers.pipelines.pt_utils import KeyDataset
from utils import parse_arguments#,calculate_classification_metrics
from peft import PeftModel
print("import library finished")

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
with open(args.transformed_squad_eval_set_path) as obj:
    file = json.loads(obj.read())[:30]

# all_data = file["data"]#["paragraphs"]["qas"]
res_dict = {}

prompt = """Write a response that appropriately completes answer the question, follow the examples. You should answer 'no answer found' if you cannot find the answer from the context.

### Example 1:
Context: 
A problem is regarded as inherently difficult if its solution requires significant resources, whatever the algorithm used. The theory formalizes this intuition, by introducing mathematical models of computation to study these problems and quantifying the amount of resources needed to solve them, such as time and storage.

Question:
What method is used to intuitively assess or quantify the amount of resources required to solve a computational problem?

Answer:
mathematical models of computation

### Example 2:
Context: 
Under the terms of the Scotland Act 1978, an elected assembly would be set up in Edinburgh provided that the majority of the Scottish electorate voted for it in a referendum to be held on 1 March 1979 that represented at least 40% of the total electorate. The 1979 Scottish devolution referendum to establish a devolved Scottish Assembly failed.

Question:
President Wilson committed his government to what in 1974?

Answer:
no answer found

### Task:
Context:
{context}

Question:
{question}

Answer:"""

for each_row in tqdm(file):
    full_prompt = prompt.format(question=each_row['question'], context=each_row['context'])
    result = pipeline(full_prompt)
    truncated = result[0]['generated_text'][len(full_prompt):].strip()
    answer = truncated

    if "no answer" in truncated.lower() or len(answer)<=3:
        answer = ""

    res_dict[each_row['id']] = answer

with open(args.squad_response_gen_file,'w') as obj:
    obj.write(json.dumps(res_dict))

#os.system(f"python scripts/official_squad_eval.py --data_file='{args.original_squad_eval_set_path}' --pred_file='{args.squad_response_gen_file}' --out-file={args.squad_eval_result_path}")
