import torch

from peft import LoraConfig
from transformers import BitsAndBytesConfig

# model name
base_model_name = "meta-llama/Llama-2-7b-hf"
# constants
TEST_SET_RATIO = 0.1
VAL_SET_RATIO = 0.2
RANDOM_SEED = 42
CUTOFF_LEN = 350
# hyperparameters
hyperparameters = {
    "temperature": 0.1,
    "num_beams": 1,
    "top_p": 0.3,
    "no_repeat_ngram_size": 3,
    "max_len": 256
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
