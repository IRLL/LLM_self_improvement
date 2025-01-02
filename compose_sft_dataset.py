"""
Author: Qianxi Li
Date: June 2, 2024
Description: This script processes natural instruction datasets for supervised fine-tuning.
"""

import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_instruction_data(base_root, output_path, limit=50):
    """
    Process natural instruction data for supervised fine-tuning.
    
    Args:
        base_root (str): Root directory containing instruction data files
        output_path (str): Path to save processed dataset
        limit (int): Maximum number of instances per task
    """
    # Initialize data collection lists
    input_list = []
    label_list = []
    
    # Process each JSON file in directory
    for filename in os.listdir(base_root):
        if ".json" not in filename:
            continue
            
        filepath = os.path.join(base_root, filename)
        logger.info(f"Processing {filename}")
        
        # Load and parse JSON content
        with open(filepath) as f:
            content = json.loads(f.read())
        
        # Format instruction text
        instruction = f"""### Instruction:\n{content["Definition"]} {content["Emphasis & Caution"]}\n\n"""
        question = "### Answer:\n"
        
        # Process instances up to limit
        instances_limit = min(limit, len(content["Instances"]))
        for i in range(instances_limit):
            # Format task and full prompt
            task = f"""### Task:\n{content["Instances"][i]["input"]}\n\n"""
            full_prompt = f"{instruction}{task}{question}"
            
            # Process label
            label = content["Instances"][i]["output"]
            if isinstance(label, list):
                label = label[0]
            
            # Verify label type
            assert isinstance(label, str), "Label must be a string type"
            
            # Add to collections
            input_list.append(full_prompt)
            label_list.append(label)
    
    # Log processing results
    logger.info(f"Processed {len(input_list)} total instances")
    
    # Save processed dataset
    output_data = {"input": input_list, "label": label_list}
    with open(output_path, 'w') as f:
        json.dump(output_data, f)
    logger.info(f"Saved processed dataset to {output_path}")

if __name__ == "__main__":
    # Define paths and parameters
    base_root = "/home/qianxi/scratch/laffi/datasets/natural_instruction_v1/train"
    output_path = "/home/qianxi/scratch/laffi/datasets/natural_instruction_v1/natural_ins_train_50.json"
    
    # Process dataset
    process_instruction_data(base_root, output_path)