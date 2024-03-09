from transformers import GPT2Tokenizer, GPT2Model,AutoModelForCausalLM,AutoTokenizer
import os, torch
import transformers
import deepspeed

os.environ["TRANSFORMERS_CACHE"] = "/home/qianxi/scratch/laffi/llama2_models"
cache_dir = "/home/qianxi/scratch/laffi/llama2_models"


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
input_text = "What is an elephant?"
model.save_pretrained("/home/qianxi/scratch/laffi/gpt2_model")
tokenizer.save_pretrained("/home/qianxi/scratch/laffi/gpt2_model")
assert 1==2
pipeline = transformers.pipeline(
    "text-generation",
    model="gpt2",
    torch_dtype=torch.float16,
    device_map="auto",
)

ds_config = {
    "fp16": {
        "enabled": True
    },
    "inference": {
        "enabled": True,
        "auto": True
    }
}
model = deepspeed.init_inference(
    AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir),
    mp_size=1,  # Set according to your model's needs and your hardware setup
    dtype=torch.float16,
    replace_method='auto',
    replace_with_kernel_inject=True,
    ds_config=ds_config
)


sequences = pipeline(
    input_text,
    do_sample=True,
    top_k=10,
    num_return_sequences=3,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")