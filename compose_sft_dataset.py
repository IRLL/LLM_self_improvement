
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader

base_root = "/home/qianxi/scratch/laffi/datasets/natural_instruction_v1/train"
limit = 50
input_list = []
label_list = []
for each_file in os.listdir(base_root):
    if ".json" in each_file:
        full_path = os.path.join(base_root, each_file)
        with open(full_path) as obj:
            content = json.loads(obj.read())

        instruction = f"""### Instruction:\n{content["Definition"]} {content["Emphasis & Caution"]}\n\n"""
        question = f"""### Answer:\n"""
        per_task_limit = limit
        if len(content["Instances"]) < per_task_limit:
            per_task_limit = len(content["Instances"])

        for i in range(per_task_limit):
            task = f"""### Task:\n{content["Instances"][i]["input"]}\n\n"""
            full_prompt = f"""{instruction}{task}{question}"""

            input_list.append(full_prompt)
            label = content["Instances"][i]["output"]

            if isinstance(label, list):
                label = content["Instances"][i]['output'][0]

            assert isinstance(label, str),"wrong type"
            label_list.append(label)

print(len(input_list))
print(len(label_list))
with open("/home/qianxi/scratch/laffi/datasets/natural_instruction_v1/natural_ins_train_50.json",'w') as obj:
    obj.write(json.dumps({"input":input_list,"label":label_list})) 
