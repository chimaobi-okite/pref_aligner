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

def load_model(model_path):
    """
    Load the specified model's text generator.
    
    Args:
        model_name (str): Name of the model.
    
    Returns:
        tuple: text pipeline generator
    """
    try:
        generator = pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            token=hf_token,
            device_map="auto",
        )

        generator.model.config.pad_token_id = generator.model.config.eos_token_id
        return generator
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    
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