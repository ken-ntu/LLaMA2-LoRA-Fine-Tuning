import string
from itertools import zip_longest

from useful_functions import *

def remove_punctuation_preprocessor(data_point):
    
    regular_punct = list(string.punctuation)
    data_point["output"] = remove_punctuation(data_point["output"], regular_punct)
    return data_point

def nil_preprocessor(task_name, data_point):
    return data_point

def mcqa_preprocessor(task_name, data_point):
    data_point = remove_punctuation_preprocessor(data_point)

    (_, option_ids, options) = mcqa_elements(task_name, data_point["input"])

    for option_id, option in zip_longest(option_ids, options):
        if data_point["output"] == option_id:
            data_point["output"] = option
            break

    return data_point

def typo_preprocessor(task_name, data_point):
    return remove_punctuation_preprocessor(data_point)

def text_simp_preprocessor(task_name, data_point):
    
    return nil_preprocessor(task_name, data_point)

preprocessor_dict = {
    "MCQA": {
        "Default": mcqa_preprocessor,
    },
    "Open-Ended": {
        "text-simplification6000": text_simp_preprocessor,
        "Default": nil_preprocessor,
    },
    "Classification": {
        "Default": nil_preprocessor,
    },
    "Others": {
        "task088_identify_typo_verification": typo_preprocessor,
        "Default": nil_preprocessor
    }
}

def set_preprocessor(task_name):
    task_type = task_type_category(task_name)["type"]

    if task_name in preprocessor_dict[task_type].keys():
        return preprocessor_dict[task_type][task_name]
    else:
        return preprocessor_dict[task_type]["Default"]
