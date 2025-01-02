# ELITE:Iterative Large Language Models Evolution through Self-Critique

## Overview
ELITE is a comprehensive framework for fine-tuning large language models (LLMs) using iterative feedback integration. The framework supports multiple model sizes (3B, 7B, 13B parameters) and includes evaluation pipelines for various downstream tasks including Natural Instructions, BoolQ, SQuAD, and GSM8K.

## Features
- Iterative fine-tuning with feedback integration
- Support for multiple LLaMA model variants (3B, 7B, 13B)
- Efficient memory management with model quantization and LoRA adapters
- Comprehensive evaluation suite across multiple tasks
- Major voting mechanism for response selection
- Prompt optimization through clustering
- Distributed training support via DeepSpeed

## Methodology
![Overview of LaFFi](img/overview.png)
Details see [Link](https://era.library.ualberta.ca/items/f74a92ea-cce9-4bee-b7b7-c44865f296d0)

## Requirements

### Core Dependencies
- `torch>=2.0.0`  
- `transformers>=4.30.0`  
- `peft>=0.4.0`  
- `accelerate>=0.20.0`  
- `bitsandbytes>=0.40.0`  
- `trl>=0.4.7`  
- `deepspeed>=0.9.0`

### Data Processing & ML
- `numpy>=1.24.0`  
- `pandas>=1.5.0`  
- `scikit-learn>=1.2.0`  
- `torchmetrics>=0.11.0`

### Utilities
- `wandb>=0.15.0`  
- `tqdm>=4.65.0`  
- `logging>=0.5.0`  
- `json5>=0.9.0`  
- `arrow>=1.2.0`

### Optional: For Development
- `pytest>=7.0.0`  
- `black>=22.0.0`

---

## Installation
1. Create a new conda environment:
```bash
conda create -n elite python=3.11.5
conda activate elite
```
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```bash
├── compose_eval_data.py        # Evaluation data composition
├── compose_sft_dataset.py      # SFT dataset preparation
├── dataset_helpers.py          # Dataset utility classes
├── feedback_inference.py       # Feedback generation pipeline
├── finetune.py                 # Model fine-tuning implementation
├── inference_helpers.py        # Inference utility functions
├── main.py                     # Main execution pipeline
├── prompt_compose_helpers.py   # Prompt composition utilities
├── scripts/                    # Execution scripts
│   ├── submit_job_3b.sh       # 3B model submission script
│   ├── submit_job_7b.sh       # 7B model submission script
│   └── submit_job_13b.sh      # 13B model submission script
└── utils.py                    # General utility functions
```

## Usage
1. Prepare your environment and model weights
2. Configure the experiment parameters in the submission scripts
3. Submit a job using the appropriate script:
```bash
bash scripts/submit_job_7b.sh # For 7B model
```

## Evaluation Tasks
- **Natural Instructions**: General instruction following capability
- **BoolQ**: Boolean question answering
- **SQuAD**: Reading comprehension and question answering
- **GSM8K**: Mathematical reasoning

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Citation
If you use this code in your research, please cite:

Here's the preliminary result of this work. The result has been accepted by AAAI 2024 - HCRL workshop and has won the best follow up paper award (2 out of 30+ accepted papers).
```bibtex
@misc{li2023laffileveraginghybridnatural,
      title={LaFFi: Leveraging Hybrid Natural Language Feedback for Fine-tuning Language Models}, 
      author={Qianxi Li and Yingyue Cao and Jikun Kang and Tianpei Yang and Xi Chen and Jun Jin and Matthew E. Taylor},
      year={2023},
      eprint={2401.00907},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2401.00907}, 
}
```

Based on the above paper's result, we did more experiments and arrived at Qianxi's MSc thesis, see [Link](https://era.library.ualberta.ca/items/f74a92ea-cce9-4bee-b7b7-c44865f296d0) for more details.

## Acknowledgments
- This research was conducted using the compute resources of Compute Canada, at the University of Alberta - Intelligent Robot Learning Lab (IRLL), in collaboration with Alberta Machine Intelligence Institute (Amii).
- Base models provided by Meta AI (LLaMA)
- Evaluation datasets provided by Hugging Face (Natural Instructions, BoolQ, SQuAD, GSM8K)
- DeepSpeed library for distributed training
- Hugging Face Transformers and Peft libraries for model loading and fine-tuning
- WandB for experiment tracking and visualization