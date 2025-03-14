from typing import Dict, List
from tqdm import tqdm
import pandas as pd
from data.prefs import PREFS
from utils.data_utils import load_data, process_df
from utils.math_utils import extract_json_from_extractor, extract_last_boxed_answer, format_math_user_prompt, math_answer_extractor
from utils.mcq_utils import calculate_math_accuracy, format_system_prompt, get_last_part
from utils.model_utils import load_model
from utils.utils import get_messages


prefs = PREFS
prefs_len = len(prefs)
print(prefs_len)


def run_math_generation(question: str, gold_answer, generator):
    user_prompt = format_math_user_prompt(question)
    responses = []
    answers = []
    is_sames = []
    for i in range(0,prefs_len+1):
        if i == 0:
            sys_message = format_system_prompt(None)
        else:
            sys_message = format_system_prompt(prefs[i])
        res = generator(get_messages(sys_message = sys_message, user_message = user_prompt), 
                                max_new_tokens=1000, )[0]["generated_text"][2]['content']
        extraction_json, usage = math_answer_extractor(question, res, gold_answer)
        
        print(usage.prompt_tokens, usage.completion_tokens)
            
        extracted_answer, is_same = extract_json_from_extractor(extraction_json)
        profile_ans = extract_last_boxed_answer(extracted_answer) if extracted_answer else None
        responses.append(res)
        answers.append(profile_ans)
        is_sames.append(is_same)
    # print(len(responses), len(answers), len(is_sames))
    return responses, answers, is_sames

def post_process_mcq_results(result_dict:Dict, model_path, data_path, chunk):
    results_df = pd.DataFrame(result_dict)
    # Calculate accuracy for each profile
    accuracy_results = {
        f'profile_{i}_answer_accuracy': calculate_math_accuracy(results_df, f'profile_{i}_is_same') 
        for i in range(0, prefs_len + 1)
    }
    
    # Display the accuracy results
    for method, accuracy in accuracy_results.items():
        print(f"{method}: {accuracy:.2%}")
        
    model_name =get_last_part(model_path)
    data_name = get_last_part(data_path)
    results_df.to_csv(f"results/math_results/{model_name}_{data_name}_{chunk}.csv", index=False)
    return results_df

def run_math_task(chunk, chunk_size, model_path,data_path):
    
    generator = load_model(model_path)
    df = load_data(data_path)
    df = process_df(df, chunk, chunk_size)
    
    print(len(df))
    

    res_dict = {f"profile_{i}_res": [] for i in range(0, prefs_len + 1)}
    answer_dict =  {f"profile_{i}_answer": [] for i in range(0, prefs_len + 1)}
    is_same_dict =  {f"profile_{i}_is_same": [] for i in range(0, prefs_len + 1)}
    gold_options = []
    for question, solution, in tqdm(
        zip(df["problem"], df['solution'],),
        total=len(df),
        desc="Processing Rows"):
        
        gold_answer = extract_last_boxed_answer(solution)

        responses, answers, is_sames = run_math_generation(question, gold_answer, generator)
        
        for i, (res, profile_ans, is_same) in enumerate(zip(responses, answers, is_sames)):
            res_dict[f"profile_{i}_res"].append(res)
            answer_dict[f"profile_{i}_answer"].append(profile_ans)
            is_same_dict[f"profile_{i}_is_same"].append(is_same)
        # print(f"question is {question}\n solution: {solution}")
        gold_option = gold_answer
        gold_options.append(gold_option)


    result_dict = {"problem": list(df["problem"]), 
                   "solution": list(df['solution']),
                   **res_dict, **answer_dict, 
                   **is_same_dict, "gold_option": gold_options}
    
    results_df = post_process_mcq_results(result_dict, model_path, data_path, chunk)