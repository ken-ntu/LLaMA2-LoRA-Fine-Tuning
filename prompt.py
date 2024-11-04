from itertools import zip_longest
from useful_functions import *

def mcqa_prompt(task_name, instruction, input, examples=None):
    sys_msg = instruction

    question, _, options = mcqa_elements(task_name, input)

    user_prompt = f"Question: {question}\nOptions:\n" + \
        "\n".join([f". {option}" for option in options]) + \
        "\nAnswer:"

    return f"{sys_msg}\n\n{user_prompt}"

def task231_prompt(task_name, instruction, input, examples=None):
    sys_msg = instruction

    question, _, options = mcqa_elements(task_name, input)

    passage = question.split('\n Links:')[0].split('Passage:')[1].strip()
    question = question.split('Passage:')[0].split('Question: ')[1].strip()

    user_prompt = f"Passage: {passage}\nQuestion: {question}\nOptions:\n" + \
        "\n".join([f". {option}" for option in options]) + \
        "\nAnswer:"

    return f"{sys_msg}\n\n{user_prompt}"

text_simplification_shot1 = {
    "Complex": "On the January 16 episode of Friday Night SmackDown, it was announced that Swagger would defend the ECW title against Hardy in a rematch at the Royal Rumble.",
    "Simple": "In the January 16 Friday Night Smackdown show, they said that Swagger would fight Hardy again to keep the ECW title at the Royal Rumble.",
}

text_simplification_shot2 = {
    "Complex": "Some trails are designated as nature trails, and are used by people learning about the natural world.",
    "Simple": "Some trails are marked as nature trails, and are used by people learning about nature.",
}

text_simplification_shot3 = {
    "Complex": "His next work, Saturday, follows an especially eventful day in the life of a successful neurosurgeon.",
    "Simple": "His next work is about a successful neurosurgeon's busy day on Saturday."
}

def text_simplification_three_shot_prompt(task_name, instruction, input, examples=None):
    # instruction = "I want you to replace my complex sentence with simple sentence(s). Keep the meaning same, but make them simpler."
    sys_msg = instruction
    
    user_prompt = ""

    if examples != None:
        text_simplification_shots = []
        for example in examples:
            text_simplification_shots.append({
                "Complex": example["input"],
                "Simple": example["output"],
            })
    else:
        text_simplification_shots = [text_simplification_shot1, text_simplification_shot2, text_simplification_shot3]

    for shot in text_simplification_shots:
        _complex = shot["Complex"]
        _simple = shot["Simple"]
        user_prompt += f"Complex: {_complex}\nSimple: {_simple}\n\n"
    
    user_prompt += f"Complex: {input}\nSimple: "

    return f"{sys_msg}\n\n{user_prompt}" 

def general_prompt(task_name, instruction, input, examples=None):
    return f"{instruction}\n\nInput:\n{input}\nAnswer:\n"

prompt_dict = {
    "MCQA": {
	"task231_iirc_link_classification": task231_prompt,
        "Default": mcqa_prompt,
    },
    "Open-Ended": {
        "task934_turk_simplification": text_simplification_three_shot_prompt,
        "text-simplification2000": general_prompt,
	    "text-simplification2400": general_prompt,
	    "text-simplification6000": text_simplification_three_shot_prompt,
	    "text-simplification100000": general_prompt,
	    "text-simplification290000": general_prompt,
        "Default": general_prompt,
    },
    "Classification": {
        "Default": general_prompt,
    },
    "Others": {
        "Default": general_prompt
    },
}

def set_prompt(task_name):
    task_type = task_type_category(task_name)["type"]
    
    if task_name in prompt_dict[task_type].keys():
        return prompt_dict[task_type][task_name]
    else:
        return prompt_dict[task_type]["Default"]
