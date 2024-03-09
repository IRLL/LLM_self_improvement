from transformers import AutoTokenizer,AutoModelForCausalLM
import transformers
import torch, os

# add = "/home/qianxi/scratch/laffi/models/models--meta-llama--Llama-2-7b-chat-hf"
os.environ['TRANSFORMERS_CACHE'] = "/home/qianxi/scratch/laffi/llama2_models"
os.environ['HF_HOME'] = "/home/qianxi/scratch/laffi/llama2_models"
#model = "/home/qianxi/scratch/laffi/models/models--meta-llama--Llama-2-7b-chat-hf"
model_name = "meta-llama/Llama-2-70b-hf"
# model_name="/home/qianxi/scratch/laffi/llama2_models"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.save_pretrained('/home/qianxi/scratch/laffi/models/70b')
tokenizer.save_pretrained('/home/qianxi/scratch/laffi/models/70b')