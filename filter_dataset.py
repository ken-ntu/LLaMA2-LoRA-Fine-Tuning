import os
import json
import string

TASK_PATH = "natural-instructions/tasks"

task_list = os.listdir(TASK_PATH)

task_list.remove("README.md")

task_category = {}
qa_tasks = []
mcqa_tasks = []
for task_json in task_list:
    with open(os.path.join(TASK_PATH, task_json), "r", encoding = "utf-8") as f:
        data = json.load(f)
    
    if len(data["Instances"]) < 2000:
        continue
    
    category = data["Categories"][0]
    if category not in task_category.keys():
        task_category[category] = []
    task_category[category].append((task_json, len(data["Instances"])))

    if category == "Question Answering":
        def remove_punct(s, punct_list):
            for punct in punct_list:
                s = s.replace(punct, ' ')
            return s

        definition = "%s" % data["Definition"][0]
        definition = remove_punct(definition, string.punctuation)
        definition = definition.lower()
        input_example = "%s" % data["Positive Examples"][0]["input"]
        input_example = remove_punct(input_example, string.punctuation)
        input_example = input_example.lower()
        if " b " in definition or " b " in input_example:
            mcqa_tasks.append((task_json, len(data["Instances"]), data["Definition"][0]))
        else:
            qa_tasks.append((task_json, len(data["Instances"]), data["Definition"][0]))

with open("MCQA.json", "w", encoding = "utf-8") as f:
    json.dump([(e[0], e[2]) for e in mcqa_tasks], f, indent = 2, ensure_ascii = False)

with open("QA.json", "w", encoding = "utf-8") as f:
    json.dump([(e[0], e[2]) for e in qa_tasks], f, indent = 2, ensure_ascii = False)

task_category["Question Answering"] = [(e[0], e[1]) for e in qa_tasks]
task_category["MCQA"] = [(e[0], e[1]) for e in mcqa_tasks]

def compare_key(e):
    return e[1]

def sort_task(e):
    return e[0].split('_')[0].split('task')[1]

def find_openqa_by_name(file_name):
    key_words = ["generation", "generate", "translation", "transfer", "summarization", "summary", "modification", "abbreviation", "compression", "simplification", "completion", "conversion", "question_decomposition", "grammar_correction", "sql"]

    for key_word in key_words:
        if key_word in file_name:
            return True

    return False

def re_category(file_name):

    if find_openqa_by_name(file_name):
        return "Open-Ended"

    with open(os.path.join(TASK_PATH, file_name), "r", encoding = "utf-8") as f:
        data = json.load(f)

    if "generate" in data["Definition"]:
        return "Open-Ended"
    
    answers = []
    for instance in data["Instances"]:
        ans = instance["output"][0]
        
        if ans not in answers:
            answers.append(ans)
        
    if len(answers) < len(data["Instances"])/10:
        return "Classification"
    return "Others"
    
def dict_append(dic, key, value):
    if key not in dic.keys():
        dic[key] = []
    dic[key].append(value)
    return dic

for category in task_category:
    min_len = min(25, len(task_category[category]))
    task_category[category].sort(reverse=True, key=compare_key)
    filtered = [e[0] for e in task_category[category][:min_len]]
    task_category[category] = filtered
    
    if category != "MCQA":
        secondary_category = {}

        for task in task_category[category]:
            second_cate = re_category(task)
            dict_append(secondary_category, second_cate, task)
    
        task_category[category] = secondary_category

task_name = {}
for category, task_type_dict in task_category.items():
    if category == "MCQA":
        for task in task_type_dict: 
            task_name[task] = {
                "category": category,
                "type": "MCQA"
            }
    else:
        for _type_, tasks in task_type_dict.items():
            for task in tasks:
                task_name[task] = {
                    "category": category,
                    "type": _type_
                }

filtered_tasks = {
    "from_category" : task_category,
    "from_task" : task_name
}

with open("filtered_tasks.json", "w", encoding = "utf-8") as f:
    json.dump(filtered_tasks, f, indent = 2, ensure_ascii = False)
