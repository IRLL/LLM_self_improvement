
from utils import load_model,load_tokenizer
import time

"""
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            # torch_dtype=torch.float16,
            #device='auto',
            max_new_tokens=50,
            do_sample=True,
            num_return_sequences=1
            # batch_size=1

        )
        pipeline.model.config.eos_token_id = pipeline.tokenizer.eos_token_id

"""
num_return_sequences=3
model = load_model("/home/qianxi/scratch/laffi/models/7b",True)
tokenizer = load_tokenizer("/home/qianxi/scratch/laffi/models/7b")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
input_context = ["hello this is a test", "that transforms a list of sentences", "into a list of list of sentences", "in order to emulate, in this case, two batches of the same lenght", "to be tokenized by the hf tokenizer for the defined model"]


def inference(model, tokenizer, batch_input_text):
    input_ids = tokenizer(batch_input_text, return_tensors="pt", padding=True, truncation=True).to('cuda:0')

    outputs = model.generate(
        input_ids=input_ids['input_ids'], 
        do_sample=True, 
        use_cache=True, 
        num_return_sequences=3,
        max_new_tokens=100,
        attention_mask=input_ids['attention_mask'] ,
        pad_token_id=tokenizer.pad_token_id
    )
    generated_texts = []
    for i in range(len(input_context)):
        # Each input's output starts at index i * num_return_sequences
        start_idx = i * num_return_sequences
        # Slice out the sequences for the current input
        batch_generated_texts = [tokenizer.decode(outputs[j], skip_special_tokens=True) for j in range(start_idx, start_idx + num_return_sequences)]
        generated_texts.append(batch_generated_texts)

    return generated_texts

