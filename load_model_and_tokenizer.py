from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from config import *

def load_model_and_tokenizer(model_path):

    """ ====== Load Model and Tokenizer ====== """
    print("Load Model and Tokenizer ...")

    pretrained_model_name_or_path = base_model_name if model_path is None else model_path

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        device_map='auto',
        quantization_config=quant_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
    )

    """ ====== Add special token ====== """
    print("Add special token ...")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("special tokens:")
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)
    print("PAD Token", tokenizer.pad_token, tokenizer.pad_token_id)

    """ ====== Model resize token embeddings ====== """
    # Since we add tokens (eos/bos/pad/others for different cases)
    print("Model resize token embeddings ...")
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer
