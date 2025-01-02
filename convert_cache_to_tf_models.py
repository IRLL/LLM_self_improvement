"""
Author: Qianxi Li
Date: June 2, 2024
Description: This script handles the conversion and saving of transformer models and tokenizers.
"""


import os
import logging
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up environment variables for transformer cache locations."""
    # Set cache directory for transformer models
    os.environ['TRANSFORMERS_CACHE'] = "/home/qianxi/scratch/laffi/llama2_models"
    # Set Hugging Face home directory
    os.environ['HF_HOME'] = "/home/qianxi/scratch/laffi/llama2_models"

def save_model_artifacts(model_path, output_path):
    """
    Load and save transformer model artifacts.
    
    Args:
        model_path (str): Path to the source model
        output_path (str): Path to save the model artifacts
    """
    # Load tokenizer from local files
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # Save tokenizer to new location
    logger.info(f"Saving tokenizer to {output_path}")
    tokenizer.save_pretrained(output_path)

if __name__ == "__main__":
    # Initialize environment
    setup_environment()
    
    # Define paths
    model_path = "/home/qianxi/scratch/laffi/models/1_3b"
    output_path = '/home/qianxi/scratch/laffi/models/llama_1_3b'
    
    # Process model artifacts
    save_model_artifacts(model_path, output_path)