import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tuning LLM")
    parser.add_argument("--answer_dataset_path", type=str, help="Path for the answer prediction dataset")
    parser.add_argument("--feedback_dataset_path", type=str, help="Path for the feedback prediction dataset")
    parser.add_argument("--base_dataset_path", type=str, help="Path for the base dataset")

    parser.add_argument("--answer_prompt_set_path", type=str, help="Path for the answer prediction prompt dataset")
    parser.add_argument("--feedback_prompt_set_path", type=str, help="Path for the feedback prediction prompt dataset")

    parser.add_argument("--model_path", type=str, default="/home/qianxi/scratch/laffi/models/7b", help="Path for the base dataset")

    parser.add_argument("--iteration_version", type=int,required=True, help="Iteration #")
    parser.add_argument("--pos_example_amount", type=int, default=2, help="Number of positive examples for this task.")
    parser.add_argument("--neg_example_amount", type=int, default=0, help="Number of negative examples for this task.")
    parser.add_argument("--current_examples_path", type=str, required=True, help="Path for the base dataset")

    return parser.parse_args()