import argparse
import functools
import json
import logging
import os, time
import wandb 

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    # Load arguments.
    args = parse_arguments()
    wandb.init(project="laffi",
            group='official',
            settings=wandb.Settings(start_method="fork"),
            config=args)
    
    # Format the date and time as a string
    task_create_time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    experiment_root_path = os.path.join(args.experiment_root_path, task_create_time_str)

    # Create experiment root
    os.makedirs(experiment_root_path)

    per_task_data_row_amount=args.per_task_data_rows


    tokenizer = load_tokenizer(args.model_path)                     
    model = load_model(args.model_path, four_bit_quant=True, adapter_path=None)
    bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    prompt_example_dict = {}
    # Loop start.
    for iteration_version in tqdm(range(args.iteration_amount)):
        # Create folder for the current iteration.
        cur_iter_root_path = os.path.join(experiment_root_path, str(iteration_version))
        os.makedirs(cur_iter_root_path)

        feedback_dataset_path = os.path.join(cur_iter_root_path,"feedback_dataset.json")
        feedback_prompt_dataset_path = os.path.join(cur_iter_root_path,"feedback_prompts.json")
        prompt_example_dict_path = os.path.join(cur_iter_root_path,"prompt_examples.json")

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

            with open(os.path.join(cur_iter_root_path, "prompt_examples.json"),'w') as obj:
                obj.write(json.dumps(prompt_example_dict))
        else:
            answer_prompt_dataset, prompt_example_dict = construct_answer_prompts(args.base_dataset_path,
                                                                                  per_task_data_row_amount,
                                                                                  "llm",
                                                                                  prompt_example_dict)

        time1 = time.time()
        # Generate answer prediction dataset.
        answer_dataset = answer_inference(model, tokenizer, answer_prompt_dataset)

        time2 = time.time()
        # Use clustering algorithm to get new examples.
        new_example_indices_dict = get_new_examples(bert,bert_tokenizer,cur_iter_root_path, answer_dataset, args.clusters)
        
        time3 = time.time()

        # Feedback generation prompts.
        # No model deploy required in this method.
        feedback_prompt_data = construct_feedback_prompts(prompt_example_dict, 
                                                          answer_dataset)

        with open(feedback_prompt_dataset_path,"w") as obj:
            obj.write(json.dumps(feedback_prompt_data))

        print(f"Feedback generation for iter={iteration_version} started")
        time4 = time.time()
        # Generate feedback dataset.
        feedback_dataset, prompt_example_dict = feedback_inference(model, tokenizer, feedback_prompt_data, new_example_indices_dict)

        time5 = time.time()

        print(f"Feedback generation for iter={iteration_version} finished")

        with open(feedback_dataset_path,"w") as obj:
            obj.write(json.dumps(feedback_dataset))

        with open(prompt_example_dict_path,"w") as obj:
            obj.write(json.dumps(prompt_example_dict))

        model.train()
        model.config.use_cache = False
        model, rouge_result = finetune(model, tokenizer, cur_iter_root_path, feedback_dataset_path)
        model.config.use_cache = True
        time6 = time.time()
        wandb.log(rouge_result[-1],step=iteration_version)


        # evaluation section.
        if args.enable_boolq_eval:
            boolq_result = eval_boolq(model, tokenizer, 
                                    boolq_eval_path="/home/qianxi/scratch/laffi/datasets/boolq/eval_boolq.json", 
                                    boolq_eval_result_path=os.path.join(cur_iter_root_path,"boolq_eval_result.json"))
            wandb.log(boolq_result,step=iteration_version)

        time7 = time.time()
        
        if args.enable_squad_eval:
            transformed_squad_eval_set_path = "/home/qianxi/scratch/laffi/datasets/squad2/truncated_processed_eval_dataset.json" #'/home/qianxi/scratch/laffi/datasets/squad2/processed_eval_dataset.json'
            original_squad_eval_set_path = "/home/qianxi/scratch/laffi/datasets/squad2/truncated_squal_eval.json"
            squad_response_gen_file = os.path.join(cur_iter_root_path, "squad_reponse_prediction.json")
            squad_eval_result_path = os.path.join(cur_iter_root_path, "squad_eval_result.json")

            squad_result = eval_squad(model,
                                    tokenizer,
                                    transformed_squad_eval_set_path, 
                                    original_squad_eval_set_path,
                                    squad_response_gen_file,
                                    squad_eval_result_path)
            wandb.log(squad_result,step=iteration_version)

        time8 = time.time()

        time_dict = {}
        time_dict["answer_inference_time"] = time2-time1
        time_dict["clustering_time"] = time3-time2
        time_dict["fb_inference_time"] = time5-time4
        time_dict["finetune_time"] = time6-time5
        if args.enable_boolq_eval:
            time_dict["boolq_time"] = time7-time6
        if args.enable_squad_eval:
            time_dict["squad_time"] = time8-time7

        wandb.log(time_dict,step=iteration_version)

    wandb.finish()

if __name__ == "__main__":
    main()