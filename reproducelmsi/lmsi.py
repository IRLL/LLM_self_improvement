import argparse
import functools
import json
import logging
import os, time
import wandb,torch
from transformers import BertTokenizer, BertModel
from datetime import datetime
from tqdm import tqdm

from utils import load_model, load_tokenizer, parse_arguments
from lmsi_inference import answer_inference, construct_answer_prompts
from eval_math import eval_gsm8k
from lmsi_finetune import finetune
from eval_boolq import eval_boolq
from squad_evaluation import eval_squad

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def main():
    # Load arguments.
    args = parse_arguments()
    wandb_enabled=args.wandb_enabled
    if wandb_enabled:
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
    bert = BertModel.from_pretrained(
        'bert-base-uncased', output_hidden_states=True)
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Loop start.
    for iteration_version in tqdm(range(args.iteration_amount)):
        # Create folder for the current iteration.
        torch.cuda.empty_cache()
        cur_iter_root_path = os.path.join(experiment_root_path, str(iteration_version))

        fb_major_voting_path = os.path.join(
            cur_iter_root_path, "major_voting.json")
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
        answer_dataset, major_voting_log = answer_inference(model, tokenizer, answer_prompt_dataset, args.contamination, args.num_return_seq, bert, bert_tokenizer)
        torch.cuda.empty_cache()
        with open(fb_major_voting_path, "w") as obj:
            obj.write(json.dumps(major_voting_log))
        time4 = time.time()


        model.train()
        model.config.use_cache = False
        model, rouge_result = finetune(model, tokenizer, cur_iter_root_path, answer_dataset,iteration_version)
        model.config.use_cache = True
        time6 = time.time()
        if wandb_enabled:
            wandb.log(rouge_result[-1],step=iteration_version+1)

        if iteration_version == args.iteration_amount-1:
            del bert
            del bert_tokenizer
            # evaluation section.
            if args.enable_boolq_eval:
                boolq_result = eval_boolq(model, tokenizer, 
                                        boolq_eval_path="/home/qianxi/scratch/laffi/datasets/boolq/eval_boolq.json", 
                                        boolq_eval_result_path=os.path.join(cur_iter_root_path,"boolq_eval_result.json"))
                torch.cuda.empty_cache()
                if wandb_enabled:
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
                torch.cuda.empty_cache()
                if wandb_enabled:
                    wandb.log(squad_result,step=iteration_version+1)
            time8 = time.time()
            if args.enable_gsm8k_eval:
                time9 = time.time()
                math_result_path = os.path.join(cur_iter_root_path, "math.json")
                acc = eval_gsm8k(model, tokenizer, args.gsm8k_testset, gsm8k_eval_result_path=math_result_path)
                torch.cuda.empty_cache()
                if wandb_enabled:
                    wandb.log({"gsm8k":acc}, step=iteration_version+1)

                time10 = time.time()

        

        time_dict = {}
        time_dict["answer_inference_time"] = time4-time1

        time_dict["finetune_time"] = time6-time4
        if iteration_version == args.iteration_amount-1:
            if args.enable_boolq_eval:
                time_dict["boolq_time"] = time7-time6
            if args.enable_squad_eval:
                time_dict["squad_time"] = time8-time7

            if args.enable_gsm8k_eval:
                time_dict["gsm8k_time"] = time10-time9

        with open(os.path.join(cur_iter_root_path, "time_usage.json"),'w') as obj: 
            obj.write(json.dumps(time_dict))
        if wandb_enabled:
            wandb.log(time_dict,step=iteration_version+1)
    if wandb_enabled:
        wandb.finish()

if __name__ == "__main__":
    main()