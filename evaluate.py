import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
                    "--test_task",
                    type=str,
                    help="not multi_task model must!! specify the main test task")
parser.add_argument("-t",
                    "--multi_task_names",
                    type=str,
                    nargs="+",
                    #required=True,
                    help="which datasets to use")
parser.add_argument("--timing",
                    choices=["before", "after"],
                    required=True,
                    help="evaluate before/after training")
parser.add_argument("-m",
                    "--multi_task_type",
                    type=str,
                    choices=["mix", "continual", "ties", "dare_ties", "dare_linear", "mix_ref_merge", "con_ref_merge"],
                    help="identify if it is a multi-task baseline or model merging method")
parser.add_argument("--merge",
                    action="store_true",
                    help="check whether merge task",
                    )
parser.add_argument("-d",
                    "--density",
                    type=float,
                    help="specify prune density"
)
parser.add_argument("-w",
                    "--weight",
                    nargs='*',
                    type=float,
                    help="specify weights of task lists",
                    )
args = parser.parse_args()

import json
import itertools
import torch
import os
import numpy as np

from tqdm import tqdm
from src.peft import PeftModel
from transformers import GenerationConfig

from config import *
from useful_functions import *
from eval_utils import *
from postprocessors import *
from load_model_and_tokenizer import load_model_and_tokenizer
from SARI import *

""" ====== CHECK INPUT ARGUMENTS ====== """
if args.multi_task_type == None:
    if args.multi_task_names is not None:
        #raise ValueError("Expected multi-task type (mix or continual) when having multiple task names")
        pass
else:
    if args.test_task == None:
        raise ValueError("Expected evaluate task name when using multi-task model")
if args.multi_task_type in ['ties']:
    print(args.multi_task_names)
    
    if len(args.multi_task_names) != len(args.weight):
        raise ValueError("Expected task name list and weight list have the same length")

""" ====== PATH ====== """
(
    TASK_DATA_PATHS,
    FILTERED_DATASET_PATH,
    OUTPUT_DIR,
    TRAIN_SET_PATH,
    TEST_SET_PATH,
    VAL_SET_PATH,
    ADAPTER_CONFIG_PATH,
    EVAL_BEF_PATHS,
    RESULT_PATH
) = set_path_constants(args.multi_task_names, args.multi_task_type)

""" ====== TARGET EVAL TASK ====== """
test_task = args.test_task if args.test_task else args.multi_task_names[0]

""" ====== Load Model and Tokenizer ====== """
print("Load Model and Tokenizer ...")

model, tokenizer = load_model_and_tokenizer(None)
if args.merge: #args.merge_type is not None:
    adapters = args.multi_task_names 
    print(f"adapters: {adapters}")

if args.merge:
    model = PeftModel.from_pretrained(model, f"output/{adapters[0]}", adapter_name=adapters[0])#OUTPUT_DIR.split('/')[1].split('_')[0]) #.')[0]) #args.task_name[0]) #"outputs/typo", adapter_name="typo") #OUTPUT_DIR
    
    for adap in adapters[1:]:
        _ = model.load_adapter(f"output/{adap}", adapter_name=adap)#.split('_')[0]) ###
    weights = args.weight
    print(f"weights: {weights}")
    adapter_name ="merge"
    density = args.density #0.2
elif args.timing=="after":
    model = PeftModel.from_pretrained(model, OUTPUT_DIR, adapter_name=test_task)
    # print("prune")
    # model.prune_exp(adapter_name=test_task, density=args.density)

if args.multi_task_type == "ties":
    model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="ties", density=density)
    model.set_adapter("merge")
elif args.multi_task_type == "dare_ties":
    model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="dare_ties", density=density)
    model.set_adapter("merge")
elif args.multi_task_type == "dare_linear":
    model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="dare_linear", density=density)
    model.set_adapter("merge")
elif args.multi_task_type == "mix_ref_merge":
    model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="mix_ref_merge", density=density)
    model.set_adapter("merge")
elif args.multi_task_type == "con_ref_merge":
    model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="con_ref_merge", density=density)
    model.set_adapter("merge")
'''
if args.timing == "after":
    if args.merge:
        model = PeftModel.from_pretrained(model, f"output/{adapters[0]}", adapter_name=adapters[0])#OUTPUT_DIR.split('/')[1].split('_')[0]) #.')[0]) #args.task_name[0]) #"outputs/typo", adapter_name="typo") #OUTPUT_DIR
    else:
        model = PeftModel.from_pretrained(model, OUTPUT_DIR, adapter_name=test_task)
    if args.multi_task_type == "ties":
        ### Model Merge (except for the main model)
        for adap in adapters[1:]:
            _ = model.load_adapter(f"output/{adap}", adapter_name=adap)#.split('_')[0]) ###
            #_ = model.load_adapter(f"outputs/threat_classify", adapter_name="threat_classify")
            #_ = model.load_adapter("outputs/summary", adapter_name="summary")
        #adapters = ["typo","threat_classify","summary"]
        #adapters.append(args.test_task)
        #print(f"adapters: {adapters}")
        #weights = [1.0, 1.0]
        #weights.append(float(args.weight))
        weights = args.weight
        print(f"weights: {weights}")
        adapter_name ="merge"
        density = 0.2
        model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="ties", density=density)
        model.set_adapter("merge")
    elif args.multi_task_type == "dare_ties":
        for adap in adapters[1:]:
            _ = model.load_adapter(f"output/{adap}", adapter_name=adap)
            #_ = model.load_adapter(f"outputs/threat_classify", adapter_name="threat_classify")
            #_ = model.load_adapter("outputs/summary", adapter_name="summary")
        #adapters = ["typo","threat_classify","summary"]
        #adapters.append(args.test_task)
        print(f"adapters: {adapters}")
        #weights = [1.0, 1.0]
        #weights.append(float(args.weight))
        weights = args.weight
        print(f"weights: {weights}")
        adapter_name ="merge"
        density = 0.2
        model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="dare_ties", density=density)
        model.set_adapter("merge")
    elif args.multi_task_type == "dare_linear":
        for adap in adapters[1:]:
            _ = model.load_adapter(f"output/{adap}", adapter_name=adap)
            #_ = model.load_adapter(f"outputs/threat_classify", adapter_name="threat_classify")
            #_ = model.load_adapter("outputs/summary", adapter_name="summary")
        #adapters = ["typo","threat_classify","summary"]
        #adapters.append(args.test_task)
        print(f"adapters: {adapters}")
        #weights = [1.0, 1.0]
        #weights.append(float(args.weight))
        weights = args.weight
        print(f"weights: {weights}")
        adapter_name ="merge"
        density = 0.2
        model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="dare_linear", density=density)
        model.set_adapter("merge")
    elif args.multi_task_type == "mix_ref_merge":
        for adap in adapters[1:]:
            _ = model.load_adapter(f"output/{adap}", adapter_name=adap)
            #_ = model.load_adapter(f"outputs/threat_classify", adapter_name="threat_classify")
            #_ = model.load_adapter("outputs/summary", adapter_name="summary")
        #adapters = ["typo","threat_classify","summary"]
        #adapters.append(args.test_task)
        print(f"adapters: {adapters}")
        #weights = [1.0, 1.0]
        #weights.append(float(args.weight))
        weights = args.weight
        print(f"weights: {weights}")
        adapter_name ="merge"
        density = 0.2
        model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="mix_ref_merge", density=density)
        model.set_adapter("merge")
    #mix_ref_merge
    elif args.multi_task_type == "con_ref_merge":
        for adap in adapters[1:]:
            _ = model.load_adapter(f"output/{adap}", adapter_name=adap)
            #_ = model.load_adapter(f"outputs/threat_classify", adapter_name="threat_classify")
            #_ = model.load_adapter("outputs/summary", adapter_name="summary")
        #adapters = ["typo","threat_classify","summary"]
        #adapters.append(args.test_task)
        print(f"adapters: {adapters}")
        #weights = [1.0, 1.0]
        #weights.append(float(args.weight))
        weights = args.weight
        print(f"weights: {weights}")
        adapter_name ="merge"
        density = 0.2
        model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="con_ref_merge", density=density)
        model.set_adapter("merge")
'''
""" ====== Generation Config ====== """
print("Setting Generation Config ...")

generation_config = GenerationConfig(
    do_sample=True,
    temperature=hyperparameters["temperature"],
    num_beams=hyperparameters["num_beams"],
    top_p=hyperparameters["top_p"],
    no_repeat_ngram_size=hyperparameters["no_repeat_ngram_size"],
    pad_token_id=tokenizer.pad_token_id
)

""" ====== Load Test Dataset ====== """
print("Load Test Dataset ...")

with open(TEST_SET_PATH, "r", encoding = "utf-8") as f:
    test_set = json.load(f)
    test_datas = test_set[test_task]

""" ====== CHECK TASK ====== """
task_is_mcqa = check_mcqa(test_task)
task_is_text_simplification = check_text_simplification(test_task)

""" ====== Evaluation ====== """
print("Evaluation ...")

results = []
correct = 0
accumulate_sari = 0
max_iteration = min(200, len(test_datas))
for (i, test_data) in tqdm(enumerate(test_datas[:max_iteration]), total = max_iteration):
    if task_is_mcqa:
        """ ====== MCQA Evaluation ====== """
        inference_prompt = generate_prompt_inference(test_task, tokenizer, test_data)
        inference_prompt_ids = tokenizer(inference_prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].cuda()

        (_, __, options) = mcqa_elements(test_task, test_data["input"])

        losses = []
        for option in options:
            tokenized_option = tokenizer(
                option,
                truncation=True,
                max_length=CUTOFF_LEN + 1,
                padding="max_length"
            )["input_ids"]

            tokenized = tokenizer(
                inference_prompt + option + tokenizer.eos_token,
                return_tensors="pt", add_special_tokens=True)

            full_tokens = tokenized["input_ids"].cuda()
            labels = full_tokens.clone()
            labels[:, :inference_prompt_ids.size(1)] = -100

            with torch.no_grad():
                loss = model(
                    input_ids=full_tokens,
                    labels=labels,
                ).loss.detach().to(torch.float16).cpu().item()
            losses.append(loss)

        pred = options[np.argmin(losses)]
        correctness = (pred == test_data["output"])
        if correctness:
            correct += 1
        
        results.append({
            "data": test_data,
            "predict": losses,
            "response": pred,
            "correctness": correctness
        })

        """ ====== MCQA Evaluation End ====== """
    elif task_is_text_simplification:
        """ ====== TEXT SIMPLIFICATION Evaluation ====== """
        predict = evaluate(
            test_task,
            model,
            tokenizer, 
            test_data, 
            generation_config, 
            hyperparameters["max_len"], 
            verbose = False )

        response, test_data = set_postprocessor(test_task)(predict, test_data)

        sari_result = sari_score(test_data["input"], response, test_data["output"])

        accumulate_sari += sari_result["SARI"]
        
        results.append({
            "data": test_data,
            "predict": predict,
            "response": response,
            "sari_result": sari_result
        })
        """ ====== TEXT SIMPLIFICATION Evaluation End ====== """
    else:
        """ ====== General Evaluation ======= """
    
        predict = evaluate(
            test_task,
            model,
            tokenizer, 
            test_data, 
            generation_config, 
            hyperparameters["max_len"], 
            verbose = False )

        response, test_data = set_postprocessor(test_task)(predict, test_data)

        correctness = (response == test_data["output"])
        if correctness:
            correct += 1

        results.append({
            "data": test_data,
            "predict": predict,
            "response": response,
            "correctness": correctness
        })
        """ ====== General Evaluation End ====== """

if task_is_text_simplification:
    avg_sari = accumulate_sari / min(max_iteration, len(test_datas))

    print(f"\n{test_task}: \ntrain {args.timing} average sari: ", avg_sari)

    eval_res = {
        "avg_sari": avg_sari,
        "results": results
    }
else:
    print(f"\n{test_task}: \ncorrect result numbers: ", correct)

    accuracy = correct / min(max_iteration, len(test_datas))
    eval_res = {
        "accuracy": accuracy,
        "results": results
    }

    print(f"{test_task}: \ntrain {args.timing} accuracy: ", accuracy)

if args.timing == "before":
    eval_bef_result = {}
    eval_bef_result["train_bef_eval_result"] = eval_res

    print(EVAL_BEF_PATHS)

    with open(EVAL_BEF_PATHS[test_task], "w", encoding = "utf-8") as f:
        json.dump(eval_bef_result, f, indent = 2, ensure_ascii = False)

else:
    """ ====== Merge and Compare Training Before and After Evaluation Results ====== """
    def InitAnsDistribution():
        (_, option_ids, _) = mcqa_elements(test_task, eval_res["results"][0]["data"]["input"])
        dis = {}
        for option_id in option_ids:
            dis[option_id] = []
        
        return dis

    with open(EVAL_BEF_PATHS[test_task], "r", encoding = "utf-8") as f:
        eval_bef_result = json.load(f)
    
    eval_bef_res = eval_bef_result["train_bef_eval_result"]
    eval_after_res = eval_res

    results = []
    if task_is_mcqa:
        ans_distribution_compare = {
            "before": InitAnsDistribution(),
            "after": InitAnsDistribution()
        }
    
    for bef, after in itertools.zip_longest(eval_bef_res["results"], eval_after_res["results"]):
        if after == None or bef == None:
            break
        if task_is_mcqa:
            (_, option_ids, options) = mcqa_elements(test_task, bef["data"]["input"])
            for option_id, option in zip_longest(option_ids, options):
                if option == bef["data"]["output"]:
                    ans_distribution_compare["before"][option_id].append(int(bef["correctness"]))
                if option == after["data"]["output"]:
                    ans_distribution_compare["after"][option_id].append(int(after["correctness"]))
    
        if task_is_text_simplification:
            results.append({
                "data": bef["data"],
                "predict": {
                    "before": bef["predict"],
                    "after": after["predict"]
                },
                "response": {
                    "before": bef["response"],
                    "after": after["response"]
                },
                "avg_sari": {
                    "before": bef["sari_result"],
                    "after": after["sari_result"]
                },
            })
        else:
            results.append({
                "data": bef["data"],
                "predict": {
                    "before": bef["predict"],
                    "after": after["predict"]
                },
                "response": {
                    "before": bef["response"],
                    "after": after["response"]
                },
                "correctness": {
                    "before": bef["correctness"],
                    "after": after["correctness"]
                },
            })
    
    recalls_compare = {
        "before": {},
        "after": {}
    }
    if task_is_mcqa:
        for timing, distribution in ans_distribution_compare.items():
            for ans, correctness in distribution.items():
                recalls_compare[timing][ans] = np.mean(correctness)

    eval_res = {}
    if task_is_text_simplification:
        eval_res["avg_sari"] = {
            "before": eval_bef_res["avg_sari"],
            "after": eval_after_res["avg_sari"]
        }
    else:
        eval_res["accuracy"] = {
            "before": eval_bef_res["accuracy"],
            "after": eval_after_res["accuracy"]
        }

    if task_is_mcqa:
        eval_res["RStd"] = {
            "before": np.std([float(v) * 100 for v in recalls_compare["before"].values()]),
            "after": np.std([float(v) * 100 for v in recalls_compare["after"].values()])
        }
        eval_res["Recalls"] = recalls_compare

    eval_res["results"] = results
    if args.merge:
        result = {}
        result["eval_result"] = eval_res
        filenames = '_'.join([str(int(w)) for w in args.weight]) #([adap.split('_')[0] for adap in adapters])

        if not os.path.exists(f"{OUTPUT_DIR}/{test_task}"): 
            os.mkdir(f"{OUTPUT_DIR}/{test_task}")
        with open(f"{OUTPUT_DIR}/{test_task}/w_{filenames}_d_{args.density}_result.json","w", encoding = "utf-8") as f:
            json.dump(result, f, indent = 2, ensure_ascii = False)
    else:
        with open(RESULT_PATH, "r", encoding = "utf-8") as f:
            result = json.load(f)

        if args.multi_task_type is None:
            result["eval_result"] = eval_res
        else:
            result[test_task] = eval_res
        if args.density is not None:
            with open(f"{OUTPUT_DIR}/d_{args.density}_result.json", "w", encoding = "utf-8") as f:
                json.dump(result, f, indent = 2, ensure_ascii = False)
        else:
            with open(RESULT_PATH, "w", encoding = "utf-8") as f:
                json.dump(result, f, indent = 2, ensure_ascii = False)

print(f"Evaluation {args.timing} Training Finished!")
