from preprocessors import *
from prompt import *
from config import *

def generate_training_data(tokenizer, data_point):
    task_name = data_point["task_name"]
    prompt = generate_prompt_inference(task_name, tokenizer, data_point)

    user_prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding=False
    )["input_ids"]

    len_user_prompt_tokens = len(user_prompt_tokens)
    
    if "text-simplification" in task_name:
        texts = prompt + "Simple: " + data_point["output"] + tokenizer.eos_token
    else:
        texts = prompt + data_point["output"] + tokenizer.eos_token
        
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length"
    )
    
    full_tokens = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    
    return {
	    "text": texts,
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens
            + full_tokens[len_user_prompt_tokens:],
        "attention_mask": attention_mask
    }

def generate_prompt_inference(task_name, tokenizer, data_point):
    preprocessor = set_preprocessor(task_name)
    data_point = preprocessor(task_name, data_point)
    if "example" in data_point:
        return set_prompt(task_name)(
            task_name = task_name,
            instruction = data_point["instruction"],
            input = data_point["input"],
            examples = data_point["examples"]
        )
    else:
        
        return set_prompt(task_name)(
            task_name = task_name,
            instruction = data_point["instruction"],
            input = data_point["input"],
        )

def evaluate(task_name, model, tokenizer, data_point, generation_config, max_len, verbose=True):
    
    prompt = generate_prompt_inference(task_name, tokenizer, data_point)
    
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].cuda()
    
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_len,
    )
    
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        if (verbose):
            print("================= Response of Model ==================")
            print({
                "input": inputs,
		"output_id": s,
		"output": output
            })

    return output
