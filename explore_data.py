from datasets import load_dataset

dataset = load_dataset("mlabonne/guanaco-llama2-1k", split="train")
print(dataset)
