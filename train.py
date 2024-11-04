import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t",
                    "--multi_task_names",
                    type=str,
                    nargs="+",
                    required=True,
                    help="which datasets to use")
parser.add_argument("-m",
                    "--multi_task_type",
                    type=str,
                    choices=["mix", "continual"],
                    help="identify if this is a multi-task baseline and whether the type of the multi-task baseline is")
parser.add_argument("--check_point",
                    type=str
                    )
parser.add_argument("--continue_training",
                    action="store_true"
                    )
args = parser.parse_args()

import os
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, PeftModel
import re
import string
import json
import random
from datasets import load_dataset, Dataset
from tqdm import tqdm
from trl import SFTTrainer

from useful_functions import *
from eval_utils import *
from config import *
from prompt import prompt_dict
from postprocessors import postprocessor_dict
from load_model_and_tokenizer import load_model_and_tokenizer

os.environ["CUDA_VISIBLE_DEVICES"]="0"

""" ====== CHECK INPUT ARGUMENTS ====== """
print("Check Input Arguments ...")

if args.multi_task_type == None:
  if len(args.multi_task_names) > 1:
    raise ValueError("Expected multi-task type (mix or continual) when having multiple task names")

""" ====== PATH ====== """
print("Set Path Constants ...")

(
  TASK_DATA_PATHS,
  FILTERED_DATASET_PATH,
  OUTPUT_DIR,
  TRAIN_SET_PATH,
  TEST_SET_PATH,
  VAL_SET_PATH,
  ADAPTER_CONFIG_PATH,
  EVAL_BEF_PATH,
  RESULT_PATH
) = set_path_constants(args.multi_task_names, args.multi_task_type)

""" ====== Load Model and Tokenizer ====== """
print("Load Model and Tokenizer ...")

model_path = OUTPUT_DIR if args.continue_training else None

model, tokenizer = load_model_and_tokenizer(model_path)

""" ====== Tokenizer Demo ====== """
print("Tokenizer Demo ...")

""" ====== Load Dataset ====== """
print("Load Dataset and Tokenized ...")

with open(TRAIN_SET_PATH, "r", encoding = "utf-8") as f:
    train_instances = json.load(f)
with open(VAL_SET_PATH, "r", encoding = "utf-8") as f:
    val_instances = json.load(f)

train_data = [generate_training_data(tokenizer, data) for data in train_instances[:6000]]
val_data = [generate_training_data(tokenizer, data) for data in val_instances]

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

from itertools import takewhile

print("user prompt token length", len([label for label in takewhile(lambda x: x == -100, train_data[0]["labels"])]))
print("id length", len([_id for _id in train_data[0]["input_ids"] if _id != 2]) + 1)
print("attention mask length", len([label for label in train_data[0]["attention_mask"] if label != 0]))

print(train_data[0])
""" ====== Training tokens Calculate ====== """
print("Training tokens Calculate ...")

token_count = 0
for data in train_data:
    for i in data["input_ids"]:
        if i < 0:
            break
        token_count += 1

print("total fine tuning tokens number", token_count)

""" ====== Freeze Model Parameters ====== """
print("Freeze Model Parameters ...")

for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1: #ndim = dim(): number of dimensions of tensor
    param.data = param.data.to(torch.float16)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float16)
model.lm_head = CastOutputToFloat(model.lm_head)

""" ====== Add Adapter layer (LoRA) ====== """
print("Add Adapter layer (LoRA) ...")

model = get_peft_model(model, lora_config)
print_trainable_parameters(model)
print(model.targeted_module_names)

""" ====== NaN solution ====== """
print("NaN solution ...")

# torch.autograd.set_detect_anomaly(True)

""" ====== Training ====== """
print("Training ...")

warmup_steps = 25
num_train_epochs = 1
learning_rate = 5e-4
max_steps = 375

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=warmup_steps,
    num_train_epochs=num_train_epochs,
    learning_rate=learning_rate,
    fp16=True,
    logging_steps=20,
    output_dir=OUTPUT_DIR,
    optim="paged_adamw_32bit",
    # evaluation_strategy="steps",
    # eval_steps=3000,
    seed=RANDOM_SEED,
    # max_steps=max_steps
)


trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

if args.check_point:
  trainer.train(args.check_point)
else:
  trainer.train()

model.save_pretrained(OUTPUT_DIR)

""" ====== Record Results ====== """
print("Record Results ...")

with open(ADAPTER_CONFIG_PATH, "r", encoding = "utf-8") as f:
  adapter_config = json.load(f)

result = adapter_config
result["task_names"] = str(args.multi_task_names)
result["TEST_SET_RATIO"] = TEST_SET_RATIO
result["VAL_SET_RATIO"] = VAL_SET_RATIO
result["RANDOM_SEED"] = RANDOM_SEED
result["warmup_steps"] = warmup_steps
result["num_train_epochs"] = num_train_epochs
result["max_steps"] = max_steps
result["learning_rate"] = learning_rate
result["trainable_parameters"] = print_trainable_parameters(model)
result["hyperparameters"] = hyperparameters
result["log_history"] = trainer.state.log_history

with open(RESULT_PATH, "w", encoding = "utf-8") as f:
    json.dump(result, f, indent = 2, ensure_ascii = False)

print("Training Stage Finished!")
