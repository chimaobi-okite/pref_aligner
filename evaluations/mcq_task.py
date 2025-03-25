from math import ceil
import os
import random
import string
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

from data.prefs import PREFS
from enums import PrefType
from preferences.prefs import CQA_PREFS, IRREVANT_PREFS, MMLU_PREFS, TQA_PREFS
from utils.data_utils import load_data, process_df
from utils.janus_utils import apply_template_mistral_instruct, extract_after_inst
from utils.mcq_extractor_utils import extract_answer_letter, extract_answer_letters_batch
from utils.mcq_utils import calculate_accuracy, format_mcq_user_prompt, format_system_prompt, get_answer_letter_option, get_last_part
from utils.model_utils import load_janus_model, load_model
from utils.utils import get_messages




irrevant_prefs = IRREVANT_PREFS
SEED = 42
random.seed(SEED)

def save_csv(df:pd.DataFrame, pref_type: str, data_name:str, model_name:str, chunk: int):
    save_path = f"results/mcq_results/{pref_type}/{data_name}"
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(f"{save_path}/{model_name}_{chunk}.csv", index=False)

def run_mcq_generation(question: str, options: List, generator, prefs: Dict, prefs_len: int):
    user_prompt = format_mcq_user_prompt(question, options)
    responses = []
    predictions = []
    for i in range(0,prefs_len+1):
        if i == 0:
            sys_message = format_system_prompt(None)
        else:
            sys_message = format_system_prompt(prefs[i])
        res = generator(get_messages(sys_message = sys_message, user_message = user_prompt), 
                                max_new_tokens=500, )[0]["generated_text"][2]['content']
        profile_ans = extract_answer_letter(res, options)
        responses.append(res)
        predictions.append(profile_ans)
    return responses, predictions

def run_janus_mcq_generation(question: str, options: List, janus, janus_tokenizer,
                             prefs: Dict, prefs_len: int):
    user_prompt = format_mcq_user_prompt(question, options)
    responses = []
    predictions = []
    system_messages = []
    for i in range(0,prefs_len+1):
        if i == 0:
            sys_message = format_system_prompt(None)
        else:
            sys_message = format_system_prompt(prefs[i])
        system_messages.append(sys_message)
            
    input_strs = [apply_template_mistral_instruct(sys_message,user_prompt) for sys_message in system_messages]
    inputs = janus_tokenizer(input_strs, return_tensors="pt", padding=True, truncation=True).to(janus.device)
    output_ids = janus.generate(inputs['input_ids'], max_new_tokens=1024, temperature=0.0,do_sample=False,
                                pad_token_id=janus_tokenizer.eos_token_id)
    responses = janus_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    responses = [extract_after_inst(res) for res in responses]
    list_of_references = [options for i in range(0,prefs_len+1)]
    predictions = extract_answer_letters_batch(responses, list_of_references)
    return responses, predictions

    
def post_process_mcq_results(result_dict:Dict, model_path, data_path, chunk, prefs_len, pref_type):
    results_df = pd.DataFrame(result_dict)
    # Calculate accuracy for each profile
    accuracy_results = {
        f'profile_{i}_answer_accuracy': calculate_accuracy(results_df, f'profile_{i}_answer') 
        for i in range(0, prefs_len + 1)
    }
    
    # Display the accuracy results
    for method, accuracy in accuracy_results.items():
        print(f"{method}: {accuracy:.2%}")
        
    model_name =get_last_part(model_path)
    data_name = get_last_part(data_path)
    save_csv(df = results_df, pref_type = pref_type, data_name = data_name,
             model_name = model_name, chunk = chunk)
    # # results_df.to_csv(f"results/mcq_results/{model_name}_{data_name}_{chunk}.csv", index=False)
    # results_df.to_csv(f"results/mcq_results/pref2_{model_name}_{data_name}_{chunk}.csv", index=False)
    return results_df
        

def truthful_qa_task(chunk, chunk_size, model_path, pref_type,
                     data_path = "truthfulqa/truthful_qa"):
    
    if pref_type == PrefType.IRRELEVANT.value:
        prefs = irrevant_prefs
    else: 
        prefs = TQA_PREFS 
    prefs_len = len(prefs)
    
    print(prefs_len)
    
    
    is_janus = False
    if "janus" in model_path:
        model, tokenizer = load_janus_model(model_path=model_path)
        is_janus = True
    else:
        generator = load_model(model_path)
    df = load_data(data_path, "multiple_choice")
    df = process_df(df, chunk, chunk_size)
    
    
    res_dict = {f"profile_{i}_res": [] for i in range(0, prefs_len + 1)}
    answer_dict =  {f"profile_{i}_answer": [] for i in range(0, prefs_len + 1)}
    gold_options = []
    references = []
    
    for question, options, in tqdm(
        zip(df["question"], df['mc1_targets'],),
        total=len(df),
        desc="Processing Rows"):
        options = list(options['choices'])
        answer = options[0]
        options = random.sample(options, len(options))
        references.append(options)

        if is_janus:
            responses, predictions = run_janus_mcq_generation(question, options, janus = model, 
                                                              janus_tokenizer = tokenizer,prefs = prefs,
                                                              prefs_len = prefs_len)
        else:
            responses, predictions = run_mcq_generation(question, options, generator, prefs, prefs_len)
        
        for i, (res, profile_ans) in enumerate(zip(responses, predictions)):
            res_dict[f"profile_{i}_res"].append(res)
            answer_dict[f"profile_{i}_answer"].append(profile_ans)
            
        gold_answer = get_answer_letter_option(answer, options)
        gold_option = gold_answer
        gold_options.append(gold_option)


    result_dict = {"question": list(df['question']),
                   "options": references,
                   **res_dict, 
                   **answer_dict, 
                   "gold_option": gold_options
                   }
    results_df = post_process_mcq_results(result_dict, model_path, data_path, chunk, prefs_len, pref_type)
    

def common_sense_qa_task(chunk, chunk_size, model_path, pref_type,
                     data_path = "tau/commonsense_qa"):
    
    if pref_type == PrefType.IRRELEVANT.value:
        prefs = irrevant_prefs
    else: 
        prefs = CQA_PREFS 
    prefs_len = len(prefs)

    is_janus = False
    if "janus" in model_path:
        model, tokenizer = load_janus_model(model_path=model_path)
        is_janus = True
    else:
        generator = load_model(model_path)
        
        
    df = load_data(data_path)
    df = process_df(df, chunk, chunk_size)

    res_dict = {f"profile_{i}_res": [] for i in range(0, prefs_len + 1)}
    answer_dict =  {f"profile_{i}_answer": [] for i in range(0, prefs_len + 1)}
    gold_options = []
    references = []
    
    for question, options, answer, in tqdm(
        zip(df["question"], df["choices"], df["answerKey"]),
        total=len(df),
        desc="Processing Rows"):
        options = list(options['text'])
        references.append(options)

        if is_janus:
            responses, predictions = run_janus_mcq_generation(question, options, janus = model, 
                                                              janus_tokenizer = tokenizer,prefs = prefs,
                                                              prefs_len = prefs_len)
        else:
            responses, predictions = run_mcq_generation(question, options, generator, prefs, prefs_len)
        
        for i, (res, profile_ans) in enumerate(zip(responses, predictions)):
            res_dict[f"profile_{i}_res"].append(res)
            answer_dict[f"profile_{i}_answer"].append(profile_ans)
            

        gold_option = answer
        gold_options.append(gold_option)


    result_dict = {"question": list(df['question']),
                   "options": references,
                   **res_dict, 
                   **answer_dict, 
                   "gold_option": gold_options
                   }
    results_df = post_process_mcq_results(result_dict, model_path, data_path, chunk, prefs_len, pref_type)
    
def mmlu_task(chunk, chunk_size, model_path, pref_type,
                     data_path = "cais/mmlu"):

    
    if pref_type == PrefType.IRRELEVANT.value:
        prefs = irrevant_prefs
    else: 
        prefs = MMLU_PREFS
    prefs_len = len(prefs)
    
    
    is_janus = False
    if "janus" in model_path:
        model, tokenizer = load_janus_model(model_path=model_path)
        is_janus = True
    else:
        generator = load_model(model_path)
        
        
    df = load_data(data_path)
    df = process_df(df, chunk, chunk_size)

    res_dict = {f"profile_{i}_res": [] for i in range(0, prefs_len + 1)}
    answer_dict =  {f"profile_{i}_answer": [] for i in range(0, prefs_len + 1)}
    gold_options = []
    references = []
    
    for question, options, answer, category, in tqdm(
        zip(df["question"], df["choices"], df["answer"], df["subject"]),
        total=len(df),
        desc="Processing Rows"):
        answer = list(string.ascii_uppercase)[:4][answer]
        references.append(options)

        if is_janus:
            responses, predictions = run_janus_mcq_generation(question, options, janus = model, 
                                                              janus_tokenizer = tokenizer,prefs = prefs,
                                                              prefs_len = prefs_len)
        else:
            responses, predictions = run_mcq_generation(question, options, generator, prefs, prefs_len)
        
        for i, (res, profile_ans) in enumerate(zip(responses, predictions)):
            res_dict[f"profile_{i}_res"].append(res)
            answer_dict[f"profile_{i}_answer"].append(profile_ans)
            

        gold_option = answer
        gold_options.append(gold_option)


    result_dict = {"question": list(df['question']),
                   "category": list(df["subject"]),
                   "options": references,
                   **res_dict, 
                   **answer_dict, 
                   "gold_option": gold_options
                   }
    results_df = post_process_mcq_results(result_dict, model_path, data_path, chunk, prefs_len, pref_type)    
    

    