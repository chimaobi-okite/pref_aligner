import os
import torch
from transformers import pipeline
import re
import argparse
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from peft import PeftModel
from dotenv import load_dotenv
from huggingface_hub import login


load_dotenv()
hf_token = os.getenv("HF_HOME_TOKEN")
login(hf_token)

# huge_models = ['mistralai/Mixtral-8x7B-Instruct-v0.1',
#                "mistralai/Mixtral-8x22B-Instruct-v0.1",
#                "meta-llama/Llama-3.3-70B-Instruct"]

# def load_model(model_path):
#     try:
#         generator = pipeline(
#             "text-generation",
#             model=model_path,
#             model_kwargs={"torch_dtype": torch.bfloat16},
#             token=hf_token,
#             device_map="auto",
#         )

#         generator.model.config.pad_token_id = generator.model.config.eos_token_id
#         return generator
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return None
    
    
# def load_model(model_path, batch_size):
#     """
#     Load the specified model's text generator in 4-bit precision.
#     For big models, load in 4bit

#     Args:
#         model_name (str): Name of the model.
#         batch_size (int): Batch size for text generation.

#     Returns:
#         pipeline: Hugging Face text generation pipeline
#     """
#     #try:
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#     )

#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         quantization_config=bnb_config,
#         device_map="auto",
#         token=hf_token
#     )

#     tokenizer = AutoTokenizer.from_pretrained(
#         model_path,
#         token=hf_token
#     )

#     eos_token_id = model.config.eos_token_id or tokenizer.eos_token_id
#     model.config.pad_token_id = eos_token_id
#     tokenizer.pad_token_id = eos_token_id

#     generator = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         batch_size=batch_size,
#         device_map="auto"
#     )

#     return generator

#     # except Exception as e:
#     #     print(f"Error loading model: {e}")
#     #     return None

huge_models = {
    'mistralai/Mixtral-8x7B-Instruct-v0.1',
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "meta-llama/Llama-3.3-70B-Instruct"
}

def load_model(model_path: str, batch_size: int = 32):
    """
    Load a text generation model with optional 4-bit quantization for large models.

    Args:
        model_path (str): Model identifier from Hugging Face hub.
        batch_size (int): Generation batch size.
        hf_token (str): Hugging Face access token.

    Returns:
        transformers.pipeline: Text generation pipeline.
    """
    is_huge = model_path in huge_models

    if is_huge:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs = {"quantization_config": quant_config}
    else:
        model_kwargs = {"torch_dtype": torch.bfloat16}

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        token=hf_token,
        # force_download=True,
        **model_kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)

    # eos_token_id = model.config.eos_token_id or tokenizer.eos_token_id
    # model.config.pad_token_id = eos_token_id
    # tokenizer.pad_token_id = eos_token_id
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    # eos_token_id = model.config.eos_token_id or tokenizer.eos_token_id

    # model.config.pad_token_id = eos_token_id
    # tokenizer.pad_token_id = eos_token_id
    
    # Set pad_token_id and model config
    pad_token_id = tokenizer.pad_token_id  # this is now a valid int
    model.config.pad_token_id = pad_token_id

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device_map="auto"
    )

    
def load_mcq_extractor_model(
    model_name="mistralai/Mistral-7B-v0.3",
    lora_model_path = "okite97/mcq_evaluator_lora",
    use_cache=False,
    load_in_4bit=True,
    quant_type="nf4",
    compute_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
):

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Configure quantization options for the model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # Load the model with quantization and other configurations
    model = AutoModelForCausalLM.from_pretrained(
        lora_model_path,
        quantization_config=bnb_config,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=low_cpu_mem_usage,
        token=hf_token
    )

    # Set caching behavior based on input parameter
    model.config.use_cache = use_cache

    return model, tokenizer

def load_janus_model(model_path="kaist-ai/janus-7b"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    dtype = torch.float16
    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
    )
    
    model.eval()
    return model, tokenizer