from transformers import AutoTokenizer,AutoModelForCausalLM
import transformers
import torch
import os
# Set the new cache directory
os.environ["TRANSFORMERS_CACHE"] = "/home/qianxi/scratch/laffi/llama2_models"
cache_dir = "/home/qianxi/scratch/laffi/llama2_models"
model = "meta-llama/Llama-2-70b-hf"

tokenizer = AutoTokenizer.from_pretrained(model,cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model, cache_dir=cache_dir)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
assert 1==2
sequences = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=400,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")