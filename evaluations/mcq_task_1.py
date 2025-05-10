from math import ceil
import os
import ast
import random
import string
from typing import Dict, List
import pandas as pd
from evaluations.runs import run_client_mcq_generation, run_janus_mcq_generation, run_mcq_generation, run_qwen_generation
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
from utils.mcq_utils import build_pref_set, calculate_accuracy, format_higher_set_system_prompt, format_mcq_user_prompt, format_system_prompt, get_answer_letter_option, get_last_part
from utils.model_utils import load_janus_model, load_model, load_qwen_model
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
                 "gpt-4o-mini-2024-07-18", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
                 "google/gemma-2-27b-it"]

irrelevant_prefs = IRRELEVANT_FOR_RELEVANT_PREFS
SEED = 42
random.seed(SEED)

def save_csv(df:pd.DataFrame, pref_type: str, model_name:str, prompt_method: str = 'direct',
             acc_diff = None, robustness = None, enable_thinking=False, chunk=None):
    save_path = f"results/mcq_results/{pref_type}/{prompt_method}/full"
    os.makedirs(save_path, exist_ok=True)
    if enable_thinking and "Qwen3" in model_name:
        model_name = model_name + "_thinking"
    print(chunk)
    if chunk is not None:
        print("entered chunk")
        model_name = model_name + f"_{chunk}"
    df.to_csv(f"{save_path}/{model_name}.csv", index=False)
    



def post_process_mcq_results(result_dict:Dict, model_path, pref_type, prompt_method, enable_thinking, chunk):
    results_df = pd.DataFrame(result_dict)
    
    
    pref_acc = calculate_accuracy(results_df, f'pref_ans')
    no_pref_acc = None
    
    if prompt_method == PromptMethod.DIRECT.value:
        no_pref_acc = calculate_accuracy(results_df, f'no_pref_ans')
        
    print(f'Pref_answer_accuracy is: {pref_acc}')
    print(f'No_Pref_answer_accuracy is: {no_pref_acc}')
    
        
    model_name =get_last_part(model_path)
   
    if no_pref_acc:
        acc_diff = no_pref_acc - pref_acc
        robustness = get_robustness(results_df)
        print(f"Acc diff {acc_diff} and robustness is {robustness}")
    
    save_csv(df = results_df, pref_type = pref_type,
             model_name = model_name, prompt_method= prompt_method, enable_thinking=enable_thinking, chunk=chunk)
    # # results_df.to_csv(f"results/mcq_results/{model_name}_{data_name}_{chunk}.csv", index=False)
    # results_df.to_csv(f"results/mcq_results/pref2_{model_name}_{data_name}_{chunk}.csv", index=False)
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

def get_robustness(df):
    df = df.copy()
    df = df[df['no_pref_ans'] == df['gold_option']]
    accuracy_result = calculate_accuracy(df, "pref_ans")
    return accuracy_result  # Skips profile_0        

def full_pref_qa_task(model_path, pref_type, prompt_method,enable_thinking,
                      chunk, chunk_size):
    
    print(f"Thinking Model: {enable_thinking}")
    print(f"Running: {model_path}, {prompt_method}, {pref_type}")
    
    BATCH_SIZE = 1
    is_janus = False
    is_qwen = False
    is_client_model = False
    is_user_message_model = False
    
    if any(name in model_path for name in ["gemma", "DeepSeek-R1-Distill-Llama-70B-free"]):
        is_user_message_model = True
            
    
    if "janus" in model_path:
        model, tokenizer = load_janus_model(model_path=model_path)
        is_janus = True
    elif "Qwen3" in model_path:
        print("running Qwen")
        model, tokenizer = load_qwen_model(model_path=model_path)
        is_qwen = True
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
    if chunk is not None and chunk_size is not None:
        df = process_df(df, chunk, chunk_size)
    
    print(f"Length of dataframe is {len(df)}")
    
    no_pref_res = []
    pref_res = []
    no_pref_ans = []
    pref_ans = []
    
    no_pref_invalids = []
    pref_invalids = []
    
    all_critics = []
    all_thoughts = []
    
    
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
        
        for i in range(2):
            if i == 0:
                if prompt_method ==PromptMethod.DIRECT.value and pref_type == PrefType.RELEVANT.value:
                    sys_messages = [format_system_prompt(None) for _ in range(len(batch_preferences))]
                else:
                    no_pref_res.extend([None] * len(batch_preferences))
                    no_pref_ans.extend([None] * len(batch_preferences))
                    no_pref_invalids.extend([0] * len(batch_preferences))
                    continue
            else:
                if pref_type == PrefType.IRRELEVANT_SET.value:
                    pref_sets = [build_pref_set(irrelevant_prefs, relevant_pref) for relevant_pref in  batch_preferences]
                    sys_messages = [format_higher_set_system_prompt(pref, examples= examples, prompt_method=prompt_method)
                                    for pref, examples in zip(pref_sets, batch_examples)]
                    
                elif pref_type == PrefType.IRRELEVANT.value:
                    pref_sets = [build_pref_set(irrelevant_prefs) for relevant_pref in  batch_preferences]
                    sys_messages = [format_system_prompt(pref, examples= examples, prompt_method=prompt_method)
                                    for pref, examples in zip(pref_sets, batch_examples)]
                else:
                    sys_messages = [
                        format_system_prompt(pref, examples=examples, prompt_method=prompt_method)
                        for pref, examples in zip(batch_preferences, batch_examples)
                        ]
        
            # print(sys_messages[0])
            if is_janus:
                responses, predictions, invalids, critics, thoughts = run_janus_mcq_generation(
                    user_prompts, batch_options_list, sys_messages,
                    janus=model, janus_tokenizer=tokenizer,
                    preferences=batch_preferences, prompt_method=prompt_method
                )
                
            elif is_qwen:
                    responses, predictions, invalids, critics, thoughts = run_qwen_generation(
                        user_prompts, batch_options_list, sys_messages,
                        model=model, tokenizer=tokenizer,
                        preferences=batch_preferences, prompt_method=prompt_method, 
                        enable_thinking=enable_thinking
                    )
                
            elif is_client_model:
                responses, predictions, invalids, critics, thoughts = run_client_mcq_generation(
                    user_prompts, batch_options_list, sys_messages,
                    client=client, model=model_path,
                    preferences=batch_preferences, prompt_method=prompt_method,
                    is_user_message_model=is_user_message_model
                )
                
            else:
                responses, predictions, invalids, critics, thoughts = run_mcq_generation(
                    user_prompts, batch_options_list, sys_messages,
                    generator, preferences=batch_preferences,
                    prompt_method=prompt_method, is_user_message_model=is_user_message_model
                )
        
        
            if i == 0:
                no_pref_res.extend(responses)
                no_pref_ans.extend(predictions)
                no_pref_invalids.extend(invalids)
            else:
                pref_res.extend(responses)
                pref_ans.extend(predictions)
                pref_invalids.extend(invalids)
                all_critics.extend(critics)
                all_thoughts.extend(thoughts)  
                  
        torch.cuda.empty_cache()
                
    result_dict = {"question": questions,
                   "options": options_list,
                   "source": sources,
                #    "category": categories,
                   "no_pref_res": no_pref_res,
                   "no_pref_ans": no_pref_ans,
                   "pref_res": pref_res,
                   "pref_ans": pref_ans, 
                   "gold_option": gold_opts,
                   "preference": preferences,
                   "no_pref_invalid": no_pref_invalids,
                   "pref_invalid": pref_invalids,
                   "critics":all_critics,
                   "thoughts":all_thoughts
                   }
    results_df = post_process_mcq_results(result_dict, model_path, pref_type, prompt_method, enable_thinking,
                                          chunk)