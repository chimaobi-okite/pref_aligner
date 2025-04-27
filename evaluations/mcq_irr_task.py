from math import ceil
import os
import ast
import random
import string
from typing import Dict, List
import pandas as pd
import torch
from tqdm import tqdm
from dotenv import load_dotenv

from openai import AzureOpenAI
from openai import OpenAI
from together import Together

from data.icl_examples import CQA_ICLS, MMLU_ICLS, TQA_ICLS
from data.prefs import PREFS
from enums import PrefType, PromptMethod
from preferences.prefs import CQA_PREFS, IRRELEVANT_FOR_RELEVANT_PREFS, IRREVANT_PREFS, MMLU_PREFS, TQA_PREFS
from utils.data_utils import load_data, load_full_data, process_df
from utils.janus_utils import apply_template_mistral_instruct, extract_after_inst
from utils.mcq_extractor_utils import extract_answer_letter, extract_answer_letters_batch
from utils.mcq_utils import build_pref_set, calculate_accuracy, format_mcq_user_prompt, format_system_prompt, get_answer_letter_option, get_last_part
from utils.model_utils import load_janus_model, load_model
from utils.utils import get_messages


load_dotenv()

endpoint = os.getenv("AZUREOPENAI_ENDPOINT")  
deployment = "gpt-4o-mini"
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "REPLACE_WITH_YOUR_KEY_VALUE_HERE")
openai_api_key = os.getenv("OPENAI_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")

openai_client = OpenAI(api_key=openai_api_key)
together_client = Together(api_key=together_api_key)


client_models = ["meta-llama/Llama-3.3-70B-Instruct-Turbo",
                 "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                 "gpt-4o-mini-2024-07-18"]

irrelevant_prefs = IRRELEVANT_FOR_RELEVANT_PREFS
SEED = 42
random.seed(SEED)

def save_csv(df:pd.DataFrame, pref_type: str, model_name:str, prompt_method: str = 'direct',):
    save_path = f"results/mcq_results/{pref_type}/{prompt_method}/full"
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(f"{save_path}/{model_name}.csv", index=False)


def run_client_mcq_generation(user_prompts, batch_options_list, sys_messages, client, model):
    batched_inputs = [get_messages(sys_message=sys_message, user_message=user_prompt) 
                    for (sys_message, user_prompt) in zip(sys_messages, user_prompts)]
    responses = []
    input_len = len(batched_inputs)
    for messages in batched_inputs:
        response = client.chat.completions.create(
            model=model, # "gpt-4-32k", # model = "deployment_name".
            messages=messages,
            temperature=0,
            seed=42,
        )
        responses.append(response.choices[0].message.content)
    list_of_references = batch_options_list[:input_len]
    list_of_references = [ast.literal_eval(refs) for refs in list_of_references]
    predictions, invalids = extract_answer_letters_batch(responses, list_of_references)
    return responses, predictions, invalids

def run_mcq_generation(user_prompts, batch_options_list, sys_messages, generator):
    batched_inputs = [get_messages(sys_message=sys_message, user_message=user_prompt) 
                    for (sys_message, user_prompt) in zip(sys_messages, user_prompts)]
    results = generator(batched_inputs, max_new_tokens=500,)
    input_len = len(batched_inputs)
    responses = [results[i][0]["generated_text"][2]['content'] for i in range(0, input_len)]
    list_of_references = batch_options_list[:input_len]
    list_of_references = [ast.literal_eval(refs) for refs in list_of_references]
    predictions, invalids = extract_answer_letters_batch(responses, list_of_references)
    return responses, predictions, invalids

def run_janus_mcq_generation(user_prompts, batch_options_list, sys_messages, janus, janus_tokenizer,):
    input_strs = [apply_template_mistral_instruct(sys_message,user_prompt) for 
                  (sys_message, user_prompt) in zip(sys_messages, user_prompts)]
    input_len = len(input_strs)
    inputs = janus_tokenizer(input_strs, return_tensors="pt", padding=True, truncation=True).to(janus.device)
    output_ids = janus.generate(inputs['input_ids'], max_new_tokens=1024, temperature=0.0,do_sample=False,
                                pad_token_id=janus_tokenizer.eos_token_id)
    responses = janus_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    responses = [extract_after_inst(res) for res in responses]
    list_of_references = batch_options_list[:input_len]
    list_of_references = [ast.literal_eval(refs) for refs in list_of_references]
    predictions, invalids = extract_answer_letters_batch(responses, list_of_references)
    return responses, predictions, invalids


def post_process_mcq_results(result_dict:Dict, model_path, pref_type, prompt_method):
    results_df = pd.DataFrame(result_dict)
    # Calculate accuracy for each profile
    
    pref_acc = calculate_accuracy(results_df, f'pref_ans')
    print(f'Pref_answer_accuracy is: {pref_acc}')
    model_name =get_last_part(model_path)
    # data_name = get_last_part(data_path)
    save_csv(df = results_df, pref_type = pref_type,
             model_name = model_name, prompt_method= prompt_method)
    return results_df

def get_batch_examples(batch_sources):
    batch_examples = []
    for source in batch_sources:
        if "truthful_qa" in source:
            icls_example = TQA_ICLS
        elif "commonsense_qa" in source:
            icls_example = CQA_ICLS
        elif "mmlu" in source:
            icls_example = MMLU_ICLS
        examples = "\n\n".join(icls_example.values())
        batch_examples.append(examples)
    return batch_examples

def full_pref_qa_task(model_path, pref_type, prompt_method,):
    
    BATCH_SIZE = 32
    print(model_path)
    is_janus = False
    is_client_model = False
            
    
    if "janus" in model_path:
        model, tokenizer = load_janus_model(model_path=model_path)
        is_janus = True
    elif model_path in client_models:
        is_client_model = True
        if "gpt" in model_path:
            client = openai_client
            print("gpt")
        else:
            client = together_client
    else:
        generator = load_model(model_path)
    df = load_full_data()
    # df = df[:10]
    # df = process_df(df, chunk, chunk_size)
    
    print(f"Length of dataframe is {len(df)}")
    
    pref_res = []
    pref_ans = []
    
    pref_invalids = []
    
    questions = df["question"].tolist()
    options_list = df["options"].tolist()
    gold_opts = df["gold_option"].tolist()
    preferences = df["preference"].tolist()
    sources = df["source"].tolist()
    
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Processing Batches"):
        batch_questions = questions[i:i+BATCH_SIZE]
        batch_options_list = options_list[i:i+BATCH_SIZE]
        batch_gold_opts = gold_opts[i:i+BATCH_SIZE]
        batch_preferences = preferences[i: i+BATCH_SIZE]
        batch_sources = sources[i: i+BATCH_SIZE]
        
        batch_examples = get_batch_examples(batch_sources)
    
        user_prompts = [format_mcq_user_prompt(question, ast.literal_eval(options)) 
                            for (question, options) in zip(batch_questions, batch_options_list)]
        
        if pref_type == PrefType.IRRELEVANT_SET.value:
            pref_sets = [build_pref_set(irrelevant_prefs, relevant_pref) for relevant_pref in  batch_preferences]
            sys_messages = [format_system_prompt(pref, examples= examples, prompt_method=prompt_method)
                            for pref, examples in zip(pref_sets, batch_examples)]
        else:
            sys_messages = [format_system_prompt(pref, examples= examples, prompt_method=prompt_method)
                                for pref, examples in zip(batch_preferences, batch_examples)]
        
        # print(sys_messages)
            
        if is_janus:
            responses, predictions, invalids = run_janus_mcq_generation(user_prompts, 
                                                                batch_options_list, 
                                                                sys_messages, janus = model,
                                                                janus_tokenizer = tokenizer)
            
        elif is_client_model:
            responses, predictions, invalids = run_client_mcq_generation(user_prompts, 
                                                                            batch_options_list, 
                                                                            sys_messages, client=client,
                                                                            model=model_path)
            
        else:
            responses, predictions, invalids = run_mcq_generation(user_prompts, 
                                                                batch_options_list, 
                                                                sys_messages, generator)
        
        
        pref_res.extend(responses)
        pref_ans.extend(predictions)
        pref_invalids.extend(invalids)
                
        torch.cuda.empty_cache()
                
    result_dict = {"question": questions,
                   "options": options_list,
                   "source": sources,
                   "pref_res": pref_res,
                   "pref_ans": pref_ans, 
                   "gold_option": gold_opts,
                   "preference": preferences,
                   "pref_invalid": pref_invalids
                   }
    results_df = post_process_mcq_results(result_dict, model_path, pref_type, prompt_method)