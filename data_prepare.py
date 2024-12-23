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
                    choices=["mix", "continual", "ties", "dare_ties", "dare_linear", "mix_ref_merge", "con_ref_merge", "ties_after_prune"],
                    help="identify if this is a multi-task baseline and whether the type of the multi-task baseline is")
parser.add_argument("--after_prune",
                    action="store_true",
                    help="check whether evaluate on condition: re-train after pruning",
                    )
args = parser.parse_args()

import os
import json
import random
from itertools import zip_longest
from sklearn.model_selection import train_test_split

from config import *
from useful_functions import *

""" ====== CHECK INPUT ARGUMENTS ====== """
if args.multi_task_type == None:
  if len(args.multi_task_names) > 1:
    raise ValueError("Expected multi-task type (mix or continual) when having multiple task names")

""" ====== PATH ====== """
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

""" ====== DATASET PREPARE RESPECTIVELY ======= """
print("DATASET PREPARE RESPECTIVELY ...")

train_instances_collection = {}
val_instances_collection = {}
test_instances_collection = {}

for task_name, TASK_DATA_PATH in TASK_DATA_PATHS.items():
  """ ====== Data Load ====== """
  print("Data Load ...")

  with open(TASK_DATA_PATH, "r", encoding = "utf-8") as f:
    data_json = json.load(f)

  data_definition = data_json["Definition"][0]
  data_instances = data_json["Instances"]
  if 'task' in task_name:
    examples = data_json["Positive Examples"]

  for instance in data_instances:
    instance["task_name"] = task_name
    instance["instruction"] = data_definition
    instance["output"] = instance["output"][0]
    if 'task' in task_name:
        instance["examples"] = examples

  """ ====== Task type is MCQA ====== """
  task_is_mcqa = check_mcqa(task_name)

  """ ====== Split into train/validation/test dataset ====== """
  print("Split into training/validation/testing dataset ...")

  if task_is_mcqa:
    tmp_instances, test_instances = train_test_split(data_instances, test_size=TEST_SET_RATIO, random_state=RANDOM_SEED)
    
    tmp_ans_cnt = count_ans_distribution(task_name, tmp_instances)
    min_cnt = min(tmp_ans_cnt.values())

    train_size = int(min_cnt * (1 - VAL_SET_RATIO)) * 4

    print("split train dataset evenly...")
    tmp_data = []
    for task_data in tmp_instances:
        ans = task_data["output"][0]
        if tmp_ans_cnt[ans] > min_cnt:
            tmp_ans_cnt[ans] -= 1
        else:
            tmp_data.append(task_data)
    tmp_instances = tmp_data

    ref = []
    for task_data in tmp_instances:
      ref.append(task_data["output"][0])
    
    train_instances, val_instances, _, __ = train_test_split(tmp_instances, ref, train_size=train_size, random_state=RANDOM_SEED, stratify=ref)

  else:
    tmp_instances, test_instances = train_test_split(data_instances, test_size=TEST_SET_RATIO, random_state=RANDOM_SEED)
    train_instances, val_instances = train_test_split(tmp_instances, test_size=VAL_SET_RATIO, random_state=RANDOM_SEED)

  train_instances_collection[task_name] = train_instances
  val_instances_collection[task_name] = val_instances
  test_instances_collection[task_name] = test_instances

""" ====== MIX DATASET ====== """
test_set = test_instances_collection

if args.multi_task_type == None:
  train_set = list(train_instances_collection.values())[0]
  val_set = list(val_instances_collection.values())[0]

elif args.multi_task_type == "mix":
  def merge_lists_evenly(lists):
    # Use zip_longest to interleave the lists, filling missing values with a placeholder
    merged = [item for group in zip_longest(*lists) for item in group if item is not None]
    return merged

  train_set = merge_lists_evenly(train_instances_collection.values())
  val_set = merge_lists_evenly(val_instances_collection.values())

else:
  train_set = [item for lists in train_instances_collection.values() for i in range(3) for item in lists]
  val_set = [item for lists in val_instances_collection.values() for item in lists]

if not os.path.exists(OUTPUT_DIR): 
    os.mkdir(OUTPUT_DIR)

if args.multi_task_type not in ['ties', 'dare_ties', 'dare_linear','ties_after_prune']:
  with open(TRAIN_SET_PATH, "w", encoding = "utf-8") as f:
    json.dump(train_set, f, indent = 2, ensure_ascii = False)
  with open(VAL_SET_PATH, "w", encoding = "utf-8") as f:
    json.dump(val_set, f, indent = 2, ensure_ascii = False)

with open(TEST_SET_PATH, "w", encoding = "utf-8") as f:
  json.dump(test_set, f, indent = 2, ensure_ascii = False)

print("Data Preparation Finished!")
