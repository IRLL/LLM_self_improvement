"""
Author: Qianxi Li
Date: June 12, 2024
Description: This module handles the inference process for answer prediction using a transformer model.
"""

import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compose_eval_data(base_root, output_path, limit=300):
    """
    Compose evaluation data from natural instruction datasets.
    
    Args:
        base_root (str): Root directory containing input JSON files
        output_path (str): Path to save the processed output
        limit (int): Maximum number of instances to process per task
        
    Returns:
        None
    """
    # Initialize lists to store processed data
    input_list = []
    label_list = []
    
    # Process each JSON file in the base directory
    for each_file in os.listdir(base_root):
        if ".json" not in each_file:
            continue
            
        full_path = os.path.join(base_root, each_file)
        logger.info(f"Processing file: {each_file}")
        
        # Read and parse JSON content
        with open(full_path) as obj:
            content = json.loads(obj.read())
        
        # Format instruction text
        instruction = f"""### Instruction:\n{content["Definition"]} {content["Emphasis & Caution"]}\n\n"""
        question = f"""### Answer:\n"""
        
        # Determine number of instances to process
        per_task_limit = min(limit, len(content["Instances"]))
        
        # Process each instance up to the limit
        for i in range(per_task_limit):
            # Format task text
            task = f"""### Task:\n{content["Instances"][i]["input"]}\n\n"""
            full_prompt = f"""{instruction}{task}{question}"""
            
            # Get output label
            label = content["Instances"][i]["output"]
            if isinstance(label, list):
                label = content["Instances"][i]['output'][0]
                
            # Verify label type
            assert isinstance(label, str), "Label must be a string"
            
            # Add to collection
            input_list.append(full_prompt)
            label_list.append(label)
    
    # Log processing statistics
    logger.info(f"Processed {len(input_list)} instances")
    
    # Save processed data
    output_data = {"input": input_list, "label": label_list}
    with open(output_path, 'w') as obj:
        json.dump(output_data, obj)
    logger.info(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    base_root = "/home/qianxi/scratch/laffi/datasets/natural_instruction_v1/eval"
    output_path = "/home/qianxi/scratch/laffi/datasets/natural_instruction_v1/natural_ins_eval_official.json"
    compose_eval_data(base_root, output_path)