from math import ceil
import string
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

from data.prefs import PREFS
from utils.data_utils import load_data, process_df
from utils.mcq_extractor_utils import extract_answer_letter
from utils.mcq_utils import calculate_accuracy, format_mcq_user_prompt, format_system_prompt, get_answer_letter_option, get_last_part
from utils.model_utils import load_model
from utils.utils import get_messages


prefs = PREFS
prefs_len = len(prefs)
print(prefs_len)

def run_mcq_generation(question: str, options: List, generator):
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
    
def post_process_mcq_results(result_dict:Dict, model_path, data_path, chunk):
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
    results_df.to_csv(f"results/mcq_results/{model_name}_{data_name}_{chunk}.csv", index=False)
    return results_df
        

def truthful_qa_task(chunk, chunk_size, model_path,
                     data_path = "truthfulqa/truthful_qa"):
    
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
        references.append(options)
        # options = random.sample(options, len(options))


        responses, predictions = run_mcq_generation(question, options, generator)
        
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
    results_df = post_process_mcq_results(result_dict, model_path, data_path, chunk)
    

def common_sense_qa_task(chunk, chunk_size, model_path,
                     data_path = "tau/commonsense_qa"):

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

        responses, predictions = run_mcq_generation(question, options, generator)
        
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
    results_df = post_process_mcq_results(result_dict, model_path, data_path, chunk)
    
def mmlu_task(chunk, chunk_size, model_path,
                     data_path = "cais/mmlu"):

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

        responses, predictions = run_mcq_generation(question, options, generator)
        
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
    results_df = post_process_mcq_results(result_dict, model_path, data_path, chunk)    
    

    
