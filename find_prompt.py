import transformers
import torch
import os,tqdm,json,time
from utils import *
import random
#CUDA_VISIBLE_DEVICES=0 python find_prompt.py
input_str = """### Context:
Please refer to the instruction and task information and give your answers. You need to follow the examples we provided.

### Instruction:
In this task we ask you to write answer to a question that involves “transient v. stationary" events, i.e., the understanding of whether an event will change over time or not. For example, the sentence "he was born in the U.S." contains a stationary event since it will last forever; however, "he is hungry" contains a transient event since it will remain true for a short period of time.  Note that a lot of the questions could have more than one correct answers. We only need a single most-likely answer. Please try to keep your "answer" as simple as possible. Concise and simple "answer" is preferred over those complex and verbose ones.  

### Examples:
### Task:
Sentence: Jack played basketball after school, after which he was very tired. 
Question: Was Jack still tired the next day?

### Answer:
No.

### Task:
Sentence: He was born in China, so he went to the Embassy to apply for a U.S. Visa. 
Question: Was he born in China by the time he gets the Visa?

### Answer:
Yes.

### Task:
Sentence: Islam later emerged as the majority religion during the centuries of Ottoman rule, though a significant Christian minority remained. 
Question: Is Islam still the majority religion?

### Answer:"""


tokenizer = load_tokenizer("/home/qianxi/scratch/laffi/models/7b")                     
model = load_model("/home/qianxi/scratch/laffi/models/7b", four_bit_quant=True, adapter_path=None)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    max_new_tokens=100, 
    do_sample=True
    # num_beams=3,
    # num_return_sequences=1,
    #batch_size=1

)

res = pipeline(input_str)
output_text = res[0]['generated_text'][len(input_str):]
truncated_result = output_text.split('\n\n')[0].strip()
print(output_text)
print('---')
print(truncated_result)
