from transformers import GPT2Tokenizer, GPT2Model,AutoModelForCausalLM,AutoTokenizer
import os, torch
import transformers
import deepspeed

os.environ["TRANSFORMERS_CACHE"] = "/home/qianxi/scratch/laffi/llama2_models"
cache_dir = "/home/qianxi/scratch/laffi/llama2_models"


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

model.save_pretrained("/home/qianxi/scratch/laffi/gpt2_model")
tokenizer.save_pretrained("/home/qianxi/scratch/laffi/gpt2_model")