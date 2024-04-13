import json
import logging
import os
import time,sys
import torch

from datetime import datetime

from utils import parse_arguments,read_json,write_json

from prompt_compose_helpers import construct_answer_prompts, construct_feedback_prompts

import warnings

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`")
debug = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

device_str = ','.join([str(i) for i in range(torch.cuda.device_count())])
def main():
    # Load arguments.
    args = parse_arguments()

    experiment_root_path = args.experiment_root_path

    if not os.path.exists(experiment_root_path):
        # Create experiment root
        os.makedirs(experiment_root_path)

    iteration_version=int(args.cur_iteration)

    prompt_example_dict = {}
    # Loop start.
    # for iteration_version in tqdm(range(args.iteration_amount)):
    # Create folder for the current iteration.
    cur_iter_root_path = os.path.join(
        experiment_root_path, str(iteration_version))
    os.makedirs(cur_iter_root_path)
    
    model_adapter_save_path = os.path.join(experiment_root_path,'adapters')
    if not os.path.exists(model_adapter_save_path):
        os.makedirs(model_adapter_save_path)

    full_model_adapter_save_path = os.path.join(model_adapter_save_path,f"model{iteration_version+1}")

    os.makedirs(full_model_adapter_save_path)

    feedback_dataset_path = os.path.join(
        cur_iter_root_path, "feedback_dataset.json")
    feedback_prompt_dataset_path = os.path.join(
        cur_iter_root_path, "feedback_prompts.json")
    prompt_example_dict_path = os.path.join(
        cur_iter_root_path, "prompt_examples.json")

    new_example_indices_dict_path = os.path.join(
        cur_iter_root_path, "new_example_indices_dict.json")

    math_result_path = os.path.join(
        cur_iter_root_path, "math.json")

    fb_major_voting_path = os.path.join(
        cur_iter_root_path, "major_voting.json")

    squad_response_gen_file = os.path.join(
        cur_iter_root_path, "squad_reponse_prediction.json")
    squad_eval_result_path = os.path.join(
        cur_iter_root_path, "squad_eval_result.json")

    adapters_path = os.path.join(experiment_root_path, "adapters")

    # Set mode to evaluation.


    # Start from scratch, use human examples to create
    # answer generation prompts.
    if iteration_version == 0:
        answer_prompt_dataset, prompt_example_dict = construct_answer_prompts(args.base_dataset_path,
                                                                                args.per_task_data_rows,
                                                                                "human",
                                                                                prompt_example_dict,
                                                                                args.pos_example_amount,
                                                                                args.neg_example_amount)

        with open(os.path.join(cur_iter_root_path, "initial_prompt_examples.json"), 'w') as obj:
            obj.write(json.dumps(prompt_example_dict))
    else:
        previous_example_dict_path = os.path.join(experiment_root_path, str(iteration_version-1), "prompt_examples.json")
        prompt_example_dict = read_json(previous_example_dict_path)
        answer_prompt_dataset, prompt_example_dict = construct_answer_prompts(args.base_dataset_path,
                                                                                args.per_task_data_rows,
                                                                                "llm",
                                                                                prompt_example_dict)

    answer_prompt_dataset_path = os.path.join(cur_iter_root_path,"answer_prompts.json")
    answer_dataset_path = os.path.join(cur_iter_root_path,"answer_dataset.json")

    write_json(answer_prompt_dataset_path, answer_prompt_dataset)


    answer_inference_args_json = {"cur_iteration":args.cur_iteration,
                                  "debug":debug,
                                  "adapters_path":adapters_path,
                                  "model_path":args.model_path,
                                  "inference_batch_size":args.eval_inference_batch_size,
                                  "answer_prompts_path":answer_prompt_dataset_path,
                                  "answer_dataset_path":answer_dataset_path}

    # Use answer prompts to generate dataset.
    exit_code = os.system(f"CUDA_VISIBLE_DEVICES={device_str} python answer_inference.py '{json.dumps(answer_inference_args_json)}'")
    if exit_code != 0:
        print(f"answer inference failed with exit code {exit_code}")
        sys.exit(1)  # Exit the program with an error status

    # Use clustering algorithm to get new examples.
    example_clustering_args_json = {"debug":debug,
                                  "experiment_root_path":cur_iter_root_path,
                                  "k":args.clusters,
                                  "new_example_indices_dict_path":new_example_indices_dict_path,
                                  "answer_dataset_path":answer_dataset_path}

    exit_code = os.system(f"CUDA_VISIBLE_DEVICES={device_str} python example_clustering.py '{json.dumps(example_clustering_args_json)}'")
    if exit_code != 0:
        print(f"example clustering failed with exit code {exit_code}")
        sys.exit(1)  # Exit the program with an error status

    answer_dataset = read_json(answer_dataset_path)
    # Feedback generation prompts.
    # No model deploy required in this method.
    feedback_prompt_data = construct_feedback_prompts(prompt_example_dict, answer_dataset)

    with open(feedback_prompt_dataset_path, "w") as obj:
        obj.write(json.dumps(feedback_prompt_data))


    feedback_inference_args_json = {"cur_iteration":args.cur_iteration,
                                    "debug":debug,
                                    "num_return_seq":args.num_return_seq,
                                    "contamination":args.contamination,
                                    "adapters_path":adapters_path,
                                    "model_path":args.model_path,
                                    "inference_batch_size":args.eval_inference_batch_size,
                                    "feedback_prompts_path":feedback_prompt_dataset_path,
                                    "feedback_dataset_path":feedback_dataset_path,
                                    "current_prompt_examples_path":prompt_example_dict_path,
                                    "major_voting_save_path":fb_major_voting_path,
                                    "new_example_indices_dict_path":new_example_indices_dict_path}

    exit_code = os.system(f"CUDA_VISIBLE_DEVICES={device_str} python feedback_inference.py '{json.dumps(feedback_inference_args_json)}'")
    if exit_code != 0:
        print(f"feedback inference failed with exit code {exit_code}")
        sys.exit(1)  # Exit the program with an error status

    finetune_arguments_json = {"cur_iteration":args.cur_iteration,
                               "adapters_path":adapters_path,
                               "model_path":args.model_path,
                               "feedback_dataset_path":feedback_dataset_path,
                               "finetune_eval_data_path":args.na_ins_evalset_path,
                               "model_adapter_save_path":full_model_adapter_save_path,
                               "result_save_path":cur_iter_root_path}
    exit_code = os.system(f"CUDA_VISIBLE_DEVICES={device_str} python finetune.py '{json.dumps(finetune_arguments_json)}'")
    if exit_code != 0:
        print(f"finetune failed with exit code {exit_code}")
        sys.exit(1)  # Exit the program with an error status

    # evaluation section.
    if args.enable_boolq_eval:
        boolq_args_json = {"cur_iteration":args.cur_iteration,
                            "adapters_path":adapters_path,
                            "model_path":args.model_path,
                            "boolq_eval_result_path":os.path.join(cur_iter_root_path, "boolq_eval_result.json"),
                            "inference_batch_size":args.eval_inference_batch_size,
                            "boolq_eval_path":args.boolq_eval_path }
        exit_code = os.system(f"CUDA_VISIBLE_DEVICES={device_str} python eval_boolq.py '{json.dumps(boolq_args_json)}'")
        if exit_code != 0:
            print(f"boolq eval failed with exit code {exit_code}")
            sys.exit(1)  # Exit the program with an error status

    if args.enable_squad_eval:
        squad_args_json = {"cur_iteration":args.cur_iteration,
                            "adapters_path":adapters_path,
                            "model_path":args.model_path,
                            "squad_eval_result_path":squad_eval_result_path,
                            "squad_response_gen_file":squad_response_gen_file,
                            "inference_batch_size":args.eval_inference_batch_size,
                            "transformed_squad_eval_set_path":args.transformed_squad_eval_set_path,
                            "original_squad_eval_set_path":args.original_squad_eval_set_path }
        exit_code = os.system(f"CUDA_VISIBLE_DEVICES={device_str} python eval_squad.py '{json.dumps(squad_args_json)}'")
        if exit_code != 0:
            print(f"eval_squad failed with exit code {exit_code}")
            sys.exit(1)  # Exit the program with an error status

    
    if args.enable_gsm8k_eval:
        math_args_json = {"cur_iteration":args.cur_iteration,
                            "adapters_path":adapters_path,
                            "model_path":args.model_path,
                            "inference_batch_size":args.eval_inference_batch_size,
                            "gsm8k_eval_result_path":math_result_path,
                            "gsm8k_eval_path":args.gsm8k_testset }

        exit_code = os.system(f"CUDA_VISIBLE_DEVICES={device_str} python eval_math.py '{json.dumps(math_args_json)}'")
        if exit_code != 0:
            print(f"eval_math failed with exit code {exit_code}")
            sys.exit(1)  # Exit the program with an error status

if __name__ == "__main__":

    main()

