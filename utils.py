import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tuning LLM")
    parser.add_argument("--answer_dataset_path", type=str, help="Path for the answer prediction dataset")
    parser.add_argument("--feedback_dataset_path", type=str, help="Path for the feedback prediction dataset")
    parser.add_argument("--base_dataset_path", type=str, help="Path for the base dataset")

    parser.add_argument("--answer_prompt_set_path", type=str, help="Path for the answer prediction prompt dataset")
    parser.add_argument("--feedback_prompt_set_path", type=str, help="Path for the feedback prediction prompt dataset")

    parser.add_argument("--model_path", type=str, default="/home/qianxi/scratch/laffi/models/7b", help="Path for the base dataset")

    parser.add_argument("--iteration_version", type=int,default=0, help="Iteration #")
    parser.add_argument("--pos_example_amount", type=int, default=2, help="Number of positive examples for this task.")
    parser.add_argument("--neg_example_amount", type=int, default=0, help="Number of negative examples for this task.")
    parser.add_argument("--current_examples_path", type=str, default=None, help="Path for the base dataset")
    parser.add_argument("--adapter_path", type=str, default=None, help="Adapter path")

    # BoolQ related arguments
    parser.add_argument("--boolq_eval_path", type=str, default=None, help="Boolq eval set path")
    parser.add_argument("--boolq_eval_result_path", type=str, default=None, help="Boolq eval result path")

    # Squad related arguments
    parser.add_argument("--transformed_squad_eval_set_path", type=str, default=None, help="Trans SQuAD eval set path")
    parser.add_argument("--original_squad_eval_set_path", type=str, default=None, help="Original SQuAD eval set path")
    parser.add_argument("--squad_response_gen_file", type=str, default=None, help="squad_response_gen_file")
    parser.add_argument("--squad_eval_result_path", type=str, default=None, help="squad_eval_result_path")


    return parser.parse_args()

def calculate_classification_metrics(predictions, labels):
    # Calculate precision
    precision = precision_score(labels, predictions)
    # Calculate recall
    recall = recall_score(labels, predictions)
    # Calculate F1 score
    f1 = f1_score(labels, predictions)
    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy
    }