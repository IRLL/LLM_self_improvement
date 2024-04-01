import argparse
import functools
import json
import logging
import os, time
import wandb 

from datetime import datetime
from tqdm import tqdm

from utils import load_model, load_tokenizer, parse_arguments
from lmsi_inference import answer_inference, construct_answer_prompts

from lmsi_finetune import finetune
from eval_boolq import eval_boolq
from squad_evaluation import eval_squad

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    # Load arguments.
    args = parse_arguments()
    wandb.init(project="laffi",
            group='lmsi_reproduce',
            settings=wandb.Settings(start_method="fork"),
            config=args)
    
    # Format the date and time as a string
    task_create_time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    full_task_name = task_create_time_str

    if args.experiment_name:
        full_task_name += args.experiment_name

    experiment_root_path = os.path.join(args.experiment_root_path, full_task_name)

    # Create experiment root
    os.makedirs(experiment_root_path)

    tokenizer = load_tokenizer(args.model_path)                     
    model = load_model(args.model_path, four_bit_quant=True, adapter_path=None)

    cur_iter_root_path = os.path.join(experiment_root_path, "before_finetune")
    os.makedirs(cur_iter_root_path)
    if args.enable_boolq_eval:

        boolq_result = eval_boolq(model, tokenizer, 
                                boolq_eval_path="/home/qianxi/scratch/laffi/datasets/boolq/eval_boolq.json", 
                                boolq_eval_result_path=os.path.join(cur_iter_root_path,"boolq_eval_result.json"))
        wandb.log(boolq_result,step=0)


    
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
        wandb.log(squad_result,step=0)

    # Loop start.
    for iteration_version in tqdm(range(args.iteration_amount)):
        # Create folder for the current iteration.
        cur_iter_root_path = os.path.join(experiment_root_path, str(iteration_version))

        
        os.makedirs(cur_iter_root_path)

        # Set mode to evaluation.
        model.eval()

        # Start from scratch, use human examples to create 
        # answer generation prompts.

        answer_prompt_dataset = construct_answer_prompts(args.base_dataset_path,
                                                        args.per_task_data_rows,
                                                        args.pos_example_amount,
                                                        args.neg_example_amount)



        time1 = time.time()
        # Generate answer prediction dataset.
        answer_dataset = answer_inference(model, tokenizer, answer_prompt_dataset)

        time4 = time.time()


        model.train()
        model.config.use_cache = False
        model, rouge_result = finetune(model, tokenizer, cur_iter_root_path, answer_dataset,iteration_version)
        model.config.use_cache = True
        time6 = time.time()
        wandb.log(rouge_result[-1],step=iteration_version+1)


        # evaluation section.
        if args.enable_boolq_eval:
            boolq_result = eval_boolq(model, tokenizer, 
                                    boolq_eval_path="/home/qianxi/scratch/laffi/datasets/boolq/eval_boolq.json", 
                                    boolq_eval_result_path=os.path.join(cur_iter_root_path,"boolq_eval_result.json"))
            wandb.log(boolq_result,step=iteration_version+1)

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
            wandb.log(squad_result,step=iteration_version+1)

        time8 = time.time()

        time_dict = {}
        time_dict["answer_inference_time"] = time4-time1

        time_dict["finetune_time"] = time6-time4
        if args.enable_boolq_eval:
            time_dict["boolq_time"] = time7-time6
        if args.enable_squad_eval:
            time_dict["squad_time"] = time8-time7

        wandb.log(time_dict,step=iteration_version+1)

    wandb.finish()

if __name__ == "__main__":
    main()