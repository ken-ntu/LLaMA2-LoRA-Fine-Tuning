from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel
)
from config_tmp_unsloth import *

def load_model_and_tokenizer(model_path):

    """ ====== Load Model and Tokenizer ====== """
    print("Load Model and Tokenizer ...")
    
    if model_path is not None:
        print("Load Model from", model_path)

    pretrained_model_name_or_path = base_model_name if model_path is None else model_path
    
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        device_map='auto',
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
    )

    """ ====== Print special token ====== """
    print("Print special token ...")
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)
    print("PAD Token", tokenizer.pad_token, tokenizer.pad_token_id)
    print("Tokenizer length", len(tokenizer))

    return model, tokenizer
