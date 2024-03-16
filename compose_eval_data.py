
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader

# class NIevalDataset(Dataset):
#     def __init__(self, tokenizer, filename="data.json", max_length=256):
#         self.tokenizer = tokenizer
#         self.inputs = []
#         self.targets = []
#         self.max_length = max_length
#         with open(filename, 'r') as file:
#             data = json.load(file)
            
#             self.inputs = data["all_feedback_input_list"]
#             self.targets= data["Feedback Label"]


#     def __len__(self):
#         return len(self.inputs)

#     def __getitem__(self, idx):
#         input_encodings = self.tokenizer(self.inputs[idx], truncation=True, padding='max_length', max_length=self.max_length)
#         target_encodings = self.tokenizer(self.targets[idx], truncation=True, padding='max_length', max_length=self.max_length)

#         return {
#             'input_ids': torch.tensor(input_encodings['input_ids']),
#             'attention_mask': torch.tensor(input_encodings['attention_mask']),
#             'labels': torch.tensor(target_encodings['input_ids'])
        # }

base_root = "/home/qianxi/scratch/laffi/datasets/natural_instruction_v1/eval"
limit = 20
input_list = []
label_list = []
for each_file in os.listdir(base_root):
    if ".json" in each_file:
        full_path = os.path.join(base_root, each_file)
        with open(full_path) as obj:
            content = json.loads(obj.read())

        instruction = f"""###Instruction:\n{content["Definition"]} {content["Emphasis & Caution"]}\n\n"""
        question = f"""###Answer:\n"""
        for i in range(limit):
            task = f"""###Task:\n{content["Instances"][i]["input"]}\n\n"""
            full_prompt = f"""{instruction}{task}{question}"""

            input_list.append(full_prompt)
            label = content["Instances"][i]["output"]

            if isinstance(label, list):
                label = content["Instances"][i]['output'][0]

            assert isinstance(label, str),"wrong type"
            label_list.append(label)

print(len(input_list))
print(len(label_list))
with open("/home/qianxi/scratch/laffi/datasets/natural_instruction_v1/natural_ins_eval.json",'w') as obj:
    obj.write(json.dumps({"input":input_list,"label":label_list})) 
