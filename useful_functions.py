import json

def remove_punctuation(text, punct_list):
    for punc in punct_list:
        if punc in text:
            text = text.replace(punc, ' ')
    return text.strip()

def print_trainable_parameters(model):

    # Prints the number of trainable parameters in the model.

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

    return f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"

def multiple_replace(s, old, new):
    for i in range(len(old)):
        s = s.replace(old[i], new[i])

    return s

def set_path_constants(task_names, multi_task=None, sparse=None):
    """
    task_names :
        list of tasks names, there will be
        only 1 task in the list if multi_task is 'None', that is, single task fine-tuning
        multiple tasks in the list if multi_task is not 'None'
     
    multi_task :
        must be 'None' or 'mix' or 'continual' or 'ties' or 'dare'
        'mix' and 'continual' for multi-task baseline
        'ties' and 'dare' for model merging
    """
    TASK_DATA_PATHS = {}
    if task_names is not None:
        for task_name in task_names:
            if task_name == "text-simplification2000":
                TASK_DATA_PATHS[task_name] = "/work/ntuee262883/LLaMA2_LoRA/WikiLarge/turkcorpus.train.json"
            elif task_name == "text-simplification2400":
                TASK_DATA_PATHS[task_name] = "/work/ntuee262883/LLaMA2_LoRA/WikiLarge/turkcorpus.TrainTestMix.8refs.json"
            elif task_name == "text-simplification100000":
                TASK_DATA_PATHS[task_name] = "/work/ntuee262883/LLaMA2_LoRA/WikiLarge/wikilarge.ori.train.json"
            elif task_name == "text-simplification290000":
                TASK_DATA_PATHS[task_name] = "/work/ntuee262883/LLaMA2_LoRA/WikiLarge/wikilarge.ori.all.json"
            elif task_name == "text-simplification6000":
                TASK_DATA_PATHS[task_name] = "/work/ntuee262883/LLaMA2_LoRA/WikiLarge/wikilarge.ori.all.json"
            else:
                TASK_DATA_PATHS[task_name] = f"/work/ntuee262883/LLaMA2_LoRA/natural-instructions/tasks/{task_name}.json"
    
    FILTERED_DATASET_PATH = "/work/ntuee262883/LLaMA2_LoRA/filtered_tasks.json"
    
    if multi_task == None:
        """
        For single task fine-tuning
        """
        OUTPUT_DIR = f"output/{task_names[0]}"
    else:
        """
        For multitask baseline or model merging
        
        ex. if tasks are task001, task002, task003 and multi_task = 'mix'
            then OUTPUT_DIR = 'output/mix_task001_task002_task003
        """
        OUTPUT_DIR = f"output/{multi_task}"
        print(f"task_names: {task_names}")
        for task_name in task_names:
            task_number = task_name.split('_')[0]
            OUTPUT_DIR += f"_{task_number}"
        
    TRAIN_SET_PATH = f"{OUTPUT_DIR}/tmp_train_dataset.json"
    TEST_SET_PATH = f"{OUTPUT_DIR}/tmp_test_dataset.json"
    VAL_SET_PATH = f"{OUTPUT_DIR}/tmp_val_dataset.json"
    ADAPTER_CONFIG_PATH = f"{OUTPUT_DIR}/adapter_config.json"

    EVAL_BEF_PATHS = {}
    for task_name in task_names:
        EVAL_BEF_PATHS[task_name] = f"output/{task_name}/eval_bef.json"
    
    RESULT_PATH = f"{OUTPUT_DIR}/result.json"

    return (
        TASK_DATA_PATHS,
        FILTERED_DATASET_PATH,
        OUTPUT_DIR,
        TRAIN_SET_PATH,
        TEST_SET_PATH,
        VAL_SET_PATH,
        ADAPTER_CONFIG_PATH,
        EVAL_BEF_PATHS,
        RESULT_PATH
    )

def mcqa_format_elements(task_name):
    prefix_dict = {
        "task1286_openbookqa_question_answering": "(",
        "task231_iirc_link_classification": " "
    }
    postfix_dict = {
        "task1286_openbookqa_question_answering": ")",
        "task231_iirc_link_classification": "."
    }
    choices_dict = {
        "task1286_openbookqa_question_answering": ["A", "B", "C", "D"],
        "task231_iirc_link_classification": ["a", "b", "c", "d"]
    }

    return prefix_dict, postfix_dict, choices_dict

def count_ans_distribution(task_name, dataset):
    
    (_, _, choices_dict) = mcqa_format_elements(task_name)

    choices = choices_dict[task_name]
    ans_cnt = {}

    for choice in choices:
        ans_cnt[str(choice)] = 0

    for data in dataset:
        ans_cnt[str(data["output"][0])] += 1

    return ans_cnt

def mcqa_elements(task_name, input):

    (prefix_dict, postfix_dict, choices_dict) = mcqa_format_elements(task_name)

    prefix = prefix_dict[task_name]
    postfix = postfix_dict[task_name]
    choices = choices_dict[task_name]

    question = input.split(f"{prefix}{choices[0]}{postfix}")[0].strip()
    options = []
    for i in range(len(choices)):
        if i == len(choices) - 1:
            options.append(
                input.split(f"{prefix}{choices[i]}{postfix}")[1].strip())
        else:
            options.append(
                input.split(f"{prefix}{choices[i]}{postfix}")[1]
                    .split(f"{prefix}{choices[i+1]}{postfix}")[0].strip())
    
    return (
        question,
        choices,
        options
    )

def check_mcqa(task_name):
    ( _, FILTERED_DATASET_PATH, _, _, _, _, _, _, _ ) = set_path_constants(task_name, None)

    with open(FILTERED_DATASET_PATH, "r", encoding = "utf-8") as f:
        filtered_tasks = json.load(f)
    
    if f"{task_name}.json" in filtered_tasks["from_category"]["MCQA"]:
        return True
    return False

def check_text_simplification(task_name):
    category = task_type_category(task_name)["category"]

    if "text-simplification" in category:
        return True
    if "task934" in task_name:
        return True
    return False

def task_type_category(task_name):
    ( _, FILTERED_DATASET_PATH, _, _, _, _, _, _, _ ) = set_path_constants(task_name, None)

    with open(FILTERED_DATASET_PATH, "r", encoding = "utf-8") as f:
        filtered_tasks = json.load(f)

    if f"{task_name}.json" not in filtered_tasks["from_task"].keys():
        return {
            "category": task_name,
            "type": "Open-Ended"
        }
    
    return filtered_tasks["from_task"][f"{task_name}.json"]
