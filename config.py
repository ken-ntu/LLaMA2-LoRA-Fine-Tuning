import torch

from peft import LoraConfig
from transformers import BitsAndBytesConfig

# model name
base_model_name = "unsloth/llama-2-7b-bnb-4bit"
# constants
TEST_SET_RATIO = 0.1
VAL_SET_RATIO = 0.2
RANDOM_SEED = 42
CUTOFF_LEN = 350
# hyperparameters
hyperparameters = {
    "num_beams": 1,
    "max_len": 350
}
# lora config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
# quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)
