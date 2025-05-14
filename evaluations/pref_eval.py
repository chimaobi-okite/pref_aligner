import argparse
import ast
import glob
import os
import pandas as pd
from tqdm import tqdm
from utils.mcq_utils import format_mcq_user_prompt
from utils.pref_eval_utils import estimate_cost, extract_judgment_from_xml, get_full_df, get_pref_evaluation, preprocess_aligner_df, preprocess_df_for_pref_eval



def save_evaluation(df, df_path:str):
    base, ext = os.path.splitext(df_path)
    new_path = f"{base}_eval{ext}"
    df.to_csv(new_path, index=False)

def run_pref_eval(df_path: str):
    if 'aligner' in df_path:
        df = pd.read_csv(df_path)
        df = preprocess_aligner_df(df)
    else:
        df: pd.DataFrame = preprocess_df_for_pref_eval(df_path=df_path)
    
    print(f"Model running on is {df_path}")
    
    # if any(x in df_path.lower() for x in ['irrelevant', 'cot', 'icl', 'self_critic']):
    #     print("doing the right thing")
    #     df_path = get_full_df(df_path = df_path)
    #     df = pd.read_csv(df_path)
    
    # if 'nopref_res' not in df.columns or 'no_pref_res' not in df.columns:
    #     print("doing the right thing")
    #     df_path = get_full_df(df_path = df_path)
    #     df = pd.read_csv(df_path)

    # df = df[:5]
    explanations = []
    answers = []
    num_prompt_tokens = []
    num_completion_tokens = []
    costs = []
    
    nopref_column = 'nopref_res'
    if nopref_column not in df.columns:
        nopref_column =  'no_pref_res'
        
    
    for i , (question, options,preference, no_pref_res, pref_res) in tqdm(enumerate(
            zip(df["question"], df['options'],df['preference'] , df[nopref_column], df['pref_res'])),
            total=len(df),
            desc="Processing Rows"):
        user_prompt = format_mcq_user_prompt(question, ast.literal_eval(options))
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
    
    df = df.rename(columns={"pref_ans": "pref_answer",
                            "no_pref_ans": "nopref_answer"
                            })
    
    if 'pref_correct' not in df.columns:
        df['pref_correct'] = df['pref_answer'] == df['gold_option']
        df['nopref_correct'] = df["nopref_answer"] == df['gold_option']
        
    print(df['pref_eval_rating'].value_counts())    
    df['pref_eval_rating'] = pd.to_numeric(df['pref_eval_rating'], errors='coerce')
    df['followed_pref'] = df['pref_eval_rating'] >= 3
    df['is_robust'] = (df['pref_correct'] & df['followed_pref'])
    save_evaluation(df, df_path=df_path)
    print(df[['question', 'nopref_res', 'nopref_answer', 'pref_answer']].isnull().sum())
    return df

def main(df_path:str):
    # dir_path = f"results/mcq_results/{pref_type}/{prompt_method}/full/{model_path}"

    # csv_files = glob.glob(os.path.join(dir_path, "*.csv"))

    # if len(csv_files) == 1:
    #     data_path = csv_files[0]
    #     print(data_path)
    # else:
    #     raise ValueError(f"Expected 1 CSV file in {dir_path}, found {len(csv_files)}: {csv_files}")
    # try:
    #     df = pd.read_csv(data_path)
    # except Exception as e:
    #     raise ValueError(f"Expected 1 CSV file in {dir_path}, found {len(csv_files)}: {csv_files}") 
    run_pref_eval(df_path=df_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Main Job")
    parser.add_argument('--df_path', type=str, required=True, help="full data path to run evaluation on")
    args = parser.parse_args()
    main(args.df_path)
