from math import ceil
import re
import string
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

from data.prefs import PREFS
from utils.data_utils import load_data, process_df
from utils.mcq_utils import calculate_accuracy, get_last_part
from utils.model_utils import load_model
from utils.utils import get_messages


prefs = PREFS
prefs_len = len(prefs)
print(prefs_len)

def math_sys_prompt(pref):
    prompt = "Answer the question below, then provide your final answer after #### \
    Only use this format for the final answer to make it easy to parse and make sure the answer is a number \n"
    if pref:
        prompt += f"Here is the user preference {pref}\n Tailor your response to thier perference\n"
    prompt += f"<<user_question_here>>\n "
    return prompt


def extract_final_integer(text):
    matches = re.findall(r'(\d+)', text)
    return str(matches[-1]) if matches else None

def extract_model_math_answer(generator, question, response):
    system_prompt = f"""Your job is to extract the final answer to a question which should be an Integer from a model response. 
    Return 'None' if there is no such answer from the response. Display your extraction after #### \n
    ### Question:
    {question}
    ### Model response:
    {response}
    ### Extracted Answer:
    """
    res =  generator(system_prompt, max_new_tokens=500, )[0]["generated_text"] #[0]['content']
    answer = extract_final_integer(res)
    return answer

def extract_gms8k_answer(text):
    match = re.search(r'###\s*(\d+)', text)
    return match.group(1) if match else None

def run_math_generation(question: str, generator):
    responses = []
    predictions = []
    for i in range(0,prefs_len+1):
        if i == 0:
            sys_message = math_sys_prompt(None)
        else:
            sys_message = math_sys_prompt(prefs[i])
        res = generator(get_messages(sys_message = sys_message, user_message = question), 
                                max_new_tokens=500, )[0]["generated_text"][1]['content']
        profile_ans = extract_model_math_answer(generator, question, res)
        responses.append(res)
        predictions.append(profile_ans)
    return responses, predictions
    
def post_process_math_results(result_dict:Dict, model_path, data_path, chunk):
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
    results_df.to_csv(f"results/math_results/{model_name}_{data_name}_{chunk}.csv", index=False)
    return results_df
        

def gms8k_task(chunk, chunk_size, model_path,
                     data_path = "openai/gsm8k"):
    
    generator = load_model(model_path)
    df = load_data(data_path, "main")
    df = process_df(df, chunk, chunk_size)
    
    print("Entered GSM8K path")

    res_dict = {f"profile_{i}_res": [] for i in range(0, prefs_len + 1)}
    answer_dict =  {f"profile_{i}_answer": [] for i in range(0, prefs_len + 1)}
    gold_options = []
    
    for question, answer, in tqdm(
        zip(df["question"], df['answer'],),
        total=len(df),
        desc="Processing Rows"):



        responses, predictions = run_math_generation(question, generator)
        
        for i, (res, profile_ans) in enumerate(zip(responses, predictions)):
            res_dict[f"profile_{i}_res"].append(res)
            answer_dict[f"profile_{i}_answer"].append(profile_ans)
            
        gold_option = extract_gms8k_answer(answer)
        gold_options.append(gold_option)

    # print(result_dict)
    result_dict = {"question": list(df['question']),
                   **res_dict, 
                   **answer_dict, 
                   "gold_option": gold_options
                   }
    print(result_dict)
    results_df = post_process_math_results(result_dict, model_path, data_path, chunk)