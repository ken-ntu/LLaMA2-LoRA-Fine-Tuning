import string

from useful_functions import *

def nil_postprocessor(predict, data_point):
    response = predict.split('Answer:\n')[1]
    return response, data_point

def general_postprocessor(predict, data_point):
    response = predict.split('Answer:\n')[1]
    #print(f"re1: {response}")
    response = response.split()[0] #'\n')[0]
    #print(f"re2: {response}")
    response = response.split('</s>')[0].strip()
    #print(f"re3: {response}")
    response = response.lower()
    #print(f"re4: {response}")
    data_point["output"] = data_point["output"].lower()
    #print(f'test data_point: {data_point["output"]}')
    return response, data_point

def language_classification_postprocessor(predict, data_point):
    response, data_point = general_postprocessor(predict, data_point)

    regular_punct = list(string.punctuation)
    response = remove_punctuation(response, regular_punct)

    return response, data_point

def text_simplification_few_shot_postprocessor(predict, data_point):
    response = predict.split('Simple: ')[-1]
    return response, data_point

postprocessor_dict = {
    "MCQA": {
        "Default": nil_postprocessor,
    },
    "Open-Ended": {
        "task934_turk_simplification": text_simplification_few_shot_postprocessor,
        "text-simplification6000": text_simplification_few_shot_postprocessor,
        "Default": nil_postprocessor,
    },
    "Classification": {
        "task1621_menyo20k-mt_en_yo_language_identification": language_classification_postprocessor,
        "task896_miam_language_classification": language_classification_postprocessor,
        "task1370_newscomm_classification": language_classification_postprocessor,
        "task441_eng_guj_parallel_corpus_gu-en_language_identification": language_classification_postprocessor,
        "task427_hindienglish_corpora_hi-en_language_identification": language_classification_postprocessor,
        "task533_europarl_es-en_language_identification": language_classification_postprocessor,
        "task447_opus_paracrawl_classification": language_classification_postprocessor,
        "task1618_cc_alligned_classify_tel_eng": language_classification_postprocessor,
        "Default": general_postprocessor,
    },
    "Others": {
        "Default": general_postprocessor
    },
}

def set_postprocessor(task_name):
    task_type = task_type_category(task_name)["type"]

    if task_name in postprocessor_dict[task_type].keys():
        return postprocessor_dict[task_type][task_name]
    else:
        return postprocessor_dict[task_type]["Default"]
