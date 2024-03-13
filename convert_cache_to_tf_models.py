from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
import transformers
import torch, os

# add = "/home/qianxi/scratch/laffi/models/models--meta-llama--Llama-2-7b-chat-hf"
os.environ['TRANSFORMERS_CACHE'] = "/home/qianxi/scratch/laffi/llama2_models"
os.environ['HF_HOME'] = "/home/qianxi/scratch/laffi/llama2_models"
#model = "/home/qianxi/scratch/laffi/models/models--meta-llama--Llama-2-7b-chat-hf"
model_name = "meta-llama/Llama-2-70b-hf"
# model_name="/home/qianxi/scratch/laffi/llama2_models"
compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(model_name,low_cpu_mem_usage=True,quantization_config=quant_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.save_pretrained('/home/qianxi/scratch/laffi/models/70b')
tokenizer.save_pretrained('/home/qianxi/scratch/laffi/models/70b')