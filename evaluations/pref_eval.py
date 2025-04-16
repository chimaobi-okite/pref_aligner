import argparse
import ast
import glob
import os
import pandas as pd
from tqdm import tqdm
from utils.mcq_utils import format_mcq_user_prompt
from utils.pref_eval_utils import estimate_cost, extract_judgment_from_xml, get_pref_evaluation, preprocess_df_for_pref_eval



def save_evaluation(df, df_path:str):
    base, ext = os.path.splitext(df_path)
    new_path = f"{base}_eval{ext}"
    df.to_csv(new_path, index=False)

def run_pref_eval(df_path: str):
    df: pd.DataFrame = preprocess_df_for_pref_eval(df_path=df_path)

    explanations = []
    answers = []
    num_prompt_tokens = []
    num_completion_tokens = []
    costs = []
    for i , (question, options,preference, no_pref_res, pref_res) in tqdm(enumerate(
            zip(df["question"], df['options'],df['preference'] , df['nopref_res'], df['pref_res'])),
            total=len(df),
            desc="Processing Rows"):
        user_prompt = format_mcq_user_prompt(question, ast.literal_eval(options))
        print(user_prompt)
        eval_res, prompt_tokens, completion_tokens = get_pref_evaluation(user_prompt, preference, no_pref_res, pref_res)
        cost = estimate_cost(prompt_tokens, completion_tokens)
        explanation, answer = extract_judgment_from_xml(eval_res)
        explanations.append(explanation)
        answers.append(answer)
        num_prompt_tokens.append(prompt_tokens)
        num_completion_tokens.append(completion_tokens)
        costs.append(cost)
        
    df['pref_eval_explanation'] = explanations
    df['pref_eval_rating'] = answers
    df['pref_eval_prompt_tokens'] = num_prompt_tokens
    df['pref_eval_completion_tokens'] = num_completion_tokens
    df['pref_eval_cost'] = costs
    save_evaluation(df, df_path=df_path)
    return df

def main(model_path: str, pref_type: str, prompt_method: str):
    dir_path = f"results/mcq_results/{pref_type}/{prompt_method}/full/{model_path}"

    csv_files = glob.glob(os.path.join(dir_path, "*.csv"))

    if len(csv_files) == 1:
        data_path = csv_files[0]
    else:
        raise ValueError(f"Expected 1 CSV file in {dir_path}, found {len(csv_files)}: {csv_files}")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise ValueError(f"Expected 1 CSV file in {dir_path}, found {len(csv_files)}: {csv_files}") 
    run_pref_eval(df_path=data_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Main Job")
    parser.add_argument('--df_path', type=str, required=True, help="full data path to run evaluation on")
    args = parser.parse_args()
    main(args.model_path, args.pref_type, args.prompt_method)
