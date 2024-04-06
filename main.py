import argparse
import functools
import json
import logging
import os
import time
import wandb
import torch

from datetime import datetime
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from example_clustering import get_new_examples
from utils import load_model, load_tokenizer, parse_arguments
from inference_helpers import answer_inference, feedback_inference
from prompt_compose_helpers import construct_answer_prompts, construct_feedback_prompts
from finetune import finetune
from eval_boolq import eval_boolq
from squad_evaluation import eval_squad
from eval_math import eval_gsm8k

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

debug = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    # Load arguments.
    args = parse_arguments()
    wandb_enabled = args.wandb_enabled

    if wandb_enabled:
        wandb.init(project="laffi",
                group='official',
                settings=wandb.Settings(start_method="fork"),
                config=args)

    # Format the date and time as a string
    task_create_time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    full_task_name = task_create_time_str

    if args.experiment_name:
        full_task_name = args.experiment_name + full_task_name

    experiment_root_path = os.path.join(
        args.experiment_root_path, full_task_name)

    # Create experiment root
    os.makedirs(experiment_root_path)

    per_task_data_row_amount = args.per_task_data_rows

    tokenizer = load_tokenizer(args.model_path)
    model = load_model(args.model_path, four_bit_quant=True, adapter_path=None)

    cur_iter_root_path = os.path.join(experiment_root_path, "before_finetune")
    os.makedirs(cur_iter_root_path)
    if args.enable_boolq_eval:

        boolq_result = eval_boolq(model, tokenizer,
                                  boolq_eval_path="/home/qianxi/scratch/laffi/datasets/boolq/eval_boolq.json",
                                  boolq_eval_result_path=os.path.join(cur_iter_root_path, "boolq_eval_result.json"))
        if wandb_enabled:
            wandb.log(boolq_result, step=0)

    if args.enable_gsm8k_eval:
        math_result_path = os.path.join(cur_iter_root_path, "math.json")

        acc = eval_gsm8k(model, tokenizer, args.gsm8k_testset, gsm8k_eval_result_path=math_result_path)
        if wandb_enabled:
            wandb.log({"gsm8k":acc}, step=0)

    if args.enable_squad_eval:
        # '/home/qianxi/scratch/laffi/datasets/squad2/processed_eval_dataset.json'
        transformed_squad_eval_set_path = "/home/qianxi/scratch/laffi/datasets/squad2/truncated_processed_eval_dataset.json"
        original_squad_eval_set_path = "/home/qianxi/scratch/laffi/datasets/squad2/truncated_squal_eval.json"
        squad_response_gen_file = os.path.join(
            cur_iter_root_path, "squad_reponse_prediction.json")
        squad_eval_result_path = os.path.join(
            cur_iter_root_path, "squad_eval_result.json")

        squad_result = eval_squad(model,
                                  tokenizer,
                                  transformed_squad_eval_set_path,
                                  original_squad_eval_set_path,
                                  squad_response_gen_file,
                                  squad_eval_result_path)
        if wandb_enabled:
            wandb.log(squad_result, step=0)

    bert = BertModel.from_pretrained(
        'bert-base-uncased', output_hidden_states=True)
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    prompt_example_dict = {}
    # Loop start.
    for iteration_version in tqdm(range(args.iteration_amount)):
        # Create folder for the current iteration.
        torch.cuda.empty_cache()
        cur_iter_root_path = os.path.join(
            experiment_root_path, str(iteration_version))
        os.makedirs(cur_iter_root_path)

        feedback_dataset_path = os.path.join(
            cur_iter_root_path, "feedback_dataset.json")
        feedback_prompt_dataset_path = os.path.join(
            cur_iter_root_path, "feedback_prompts.json")
        prompt_example_dict_path = os.path.join(
            cur_iter_root_path, "prompt_examples.json")

        math_result_path = os.path.join(
            cur_iter_root_path, "math.json")

        fb_major_voting_path = os.path.join(
            cur_iter_root_path, "major_voting.json")

        # Set mode to evaluation.
        model.eval()

        # Start from scratch, use human examples to create
        # answer generation prompts.
        if iteration_version == 0:
            answer_prompt_dataset, prompt_example_dict = construct_answer_prompts(args.base_dataset_path,
                                                                                  per_task_data_row_amount,
                                                                                  "human",
                                                                                  prompt_example_dict,
                                                                                  args.pos_example_amount,
                                                                                  args.neg_example_amount)

            with open(os.path.join(cur_iter_root_path, "prompt_examples.json"), 'w') as obj:
                obj.write(json.dumps(prompt_example_dict))
        else:
            answer_prompt_dataset, prompt_example_dict = construct_answer_prompts(args.base_dataset_path,
                                                                                  per_task_data_row_amount,
                                                                                  "llm",
                                                                                  prompt_example_dict)

        time1 = time.time()
        # Generate answer prediction dataset.
        answer_dataset = answer_inference(
            model, tokenizer, answer_prompt_dataset, debug)

        time2 = time.time()

        # Use clustering algorithm to get new examples.
        new_example_indices_dict = get_new_examples(
            bert, bert_tokenizer, cur_iter_root_path, answer_dataset, args.clusters, debug)

        time3 = time.time()

        # Feedback generation prompts.
        # No model deploy required in this method.
        feedback_prompt_data = construct_feedback_prompts(prompt_example_dict,
                                                          answer_dataset)

        with open(feedback_prompt_dataset_path, "w") as obj:
            obj.write(json.dumps(feedback_prompt_data))

        print(f"Feedback generation for iter={iteration_version} started")
        time4 = time.time()
        # Generate feedback dataset.
        feedback_dataset, prompt_example_dict, major_voting_log = feedback_inference(model, 
                                                                                    tokenizer, 
                                                                                    feedback_prompt_data, 
                                                                                    new_example_indices_dict, 
                                                                                    args.num_return_seq, 
                                                                                    bert, 
                                                                                    bert_tokenizer, 
                                                                                    args.contamination,
                                                                                    debug)
        time5 = time.time()

        print(f"Feedback generation for iter={iteration_version} finished")

        with open(feedback_dataset_path, "w") as obj:
            obj.write(json.dumps(feedback_dataset))

        with open(prompt_example_dict_path, "w") as obj:
            obj.write(json.dumps(prompt_example_dict))

        with open(fb_major_voting_path, "w") as obj:
            obj.write(json.dumps(major_voting_log))

        model.train()
        model.config.use_cache = False
        model, rouge_result = finetune(
            model, tokenizer, cur_iter_root_path, feedback_dataset_path)
        model.config.use_cache = True
        time6 = time.time()
        if wandb_enabled:
            wandb.log(rouge_result[-1], step=iteration_version+1)

        # evaluation section.
        if args.enable_boolq_eval:
            boolq_result = eval_boolq(model, tokenizer,
                                      boolq_eval_path="/home/qianxi/scratch/laffi/datasets/boolq/eval_boolq.json",
                                      boolq_eval_result_path=os.path.join(cur_iter_root_path, "boolq_eval_result.json"))
            if wandb_enabled:
                wandb.log(boolq_result, step=iteration_version+1)

        time7 = time.time()

        if args.enable_squad_eval:
            # '/home/qianxi/scratch/laffi/datasets/squad2/processed_eval_dataset.json'
            transformed_squad_eval_set_path = "/home/qianxi/scratch/laffi/datasets/squad2/truncated_processed_eval_dataset.json"
            original_squad_eval_set_path = "/home/qianxi/scratch/laffi/datasets/squad2/truncated_squal_eval.json"
            squad_response_gen_file = os.path.join(
                cur_iter_root_path, "squad_reponse_prediction.json")
            squad_eval_result_path = os.path.join(
                cur_iter_root_path, "squad_eval_result.json")

            squad_result = eval_squad(model,
                                      tokenizer,
                                      transformed_squad_eval_set_path,
                                      original_squad_eval_set_path,
                                      squad_response_gen_file,
                                      squad_eval_result_path)
            if wandb_enabled:
                wandb.log(squad_result, step=iteration_version+1)

            time8 = time.time()

        
        if args.enable_gsm8k_eval:
            time9 = time.time()
            acc = eval_gsm8k(model, tokenizer, args.gsm8k_testset, gsm8k_eval_result_path=math_result_path)
            if wandb_enabled:
                wandb.log({"gsm8k":acc}, step=iteration_version+1)

            time10 = time.time()
        

        time_dict = {}
        time_dict["answer_inference_time"] = time2-time1
        time_dict["clustering_time"] = time3-time2
        time_dict["fb_inference_time"] = time5-time4
        time_dict["finetune_time"] = time6-time5
        if args.enable_boolq_eval:
            time_dict["boolq_time"] = time7-time6
        if args.enable_squad_eval:
            time_dict["squad_time"] = time8-time7
        if args.enable_gsm8k_eval:
            time_dict["gsm8k_time"] = time10-time9

        with open(os.path.join(cur_iter_root_path, "time_usage.json"),'w') as obj: 
            obj.write(json.dumps(time_dict))

        if wandb_enabled:
            wandb.log(time_dict, step=iteration_version+1)

    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
