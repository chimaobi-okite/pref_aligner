import os
import pandas as pd
from math import ceil
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import random
import string
from tqdm import tqdm

from preferences.prefs import CQA_PREFS, MMLU_PREFS, TQA_PREFS
from utils.mcq_utils import get_answer_letter_option, get_answer_text
SEED = 42

mmlu_subjects = ['professional_law', 'high_school_biology','professional_accounting', 'professional_medicine',
       'high_school_mathematics', 'high_school_microeconomics','conceptual_physics', 'marketing',
       'high_school_statistics','high_school_chemistry', 'college_medicine', 'high_school_physics',
       'electrical_engineering', 'college_biology', 'anatomy', 'formal_logic',
       'college_physics', 'college_mathematics','abstract_algebra', 'business_ethics', 'college_chemistry']

def load_data(path:str, split:str = None, sample_size: int = None):
    if "mmlu" in path:
        df = load_mmlu(path)
        if sample_size:
            df = df.sample(sample_size, random_state=SEED)
        return df
    if "lighteval-MATH" in path:
        df = load_math(path, sample_size=sample_size)
        return df
    if split:
        ds = load_dataset(path, split)
    else:
        ds = load_dataset(path)
    if "gsm8k" in path:
        df = ds['test'].to_pandas()
    else:
        df = ds['validation'].to_pandas()
    if sample_size:
        df = df.sample(sample_size, random_state=SEED)
    return df


def load_mmlu(path:str):
  ds = load_dataset(path, "all", split = "test")
  df = ds.to_pandas()
  df = df[df['subject'].isin(mmlu_subjects)]
  print("Total samples after filtering:", len(df))
  return df


def load_math(path:str, sample_size):
    data_size = 1000
    ds = load_dataset(path)
    test_df = ds["test"].to_pandas()
    stratified_sample, _ = train_test_split(test_df, stratify=test_df['level'], 
                                            train_size=data_size, random_state=42)
    if sample_size:
        stratified_sample = stratified_sample.sample(sample_size, random_state=SEED)
    return stratified_sample


def load_full_data(path = "data/robuset_main.csv", sample_size = None):
    df = pd.read_csv(path)
    # df = df[df['source'] == "cais/mmlu"]
    # # df = df[df['source'] == "truthfulqa/truthful_qa"]
    if sample_size:
        sample = df[:sample_size]
        return sample
    return df

def process_df(df, chunk, chunk_size):
    total_size = len(df)
    per_chunk_data_size = ceil(total_size / chunk_size)
    print(f"Total dataset size: {total_size}")
    
    start_index = chunk * per_chunk_data_size
    end_index = min(start_index + per_chunk_data_size, total_size)  # Ensure end_index is within bounds
    
    if start_index >= total_size:
        print("Warning: Start index exceeds dataset size. No data to process.")
        return None  # Or handle it as needed
    
    df = df[start_index:end_index]
    return df

def aggregate_datasets(tqa_df: pd.DataFrame,
                       cqa_df: pd.DataFrame,
                       mmlu_df: pd.DataFrame) -> pd.DataFrame:
    
    "Function aggregates dataframes to create our full datasets"
    data = {
        "id": [], "question": [], "options": [],
        "gold_option": [], "gold_answer": [], "category": [],
        "source": [], "preference": []
    }

    def _get_pref(prefs, idx):
        return prefs[(idx % len(prefs)) + 1]

    def append_row(idx, q, opts, gold_opt, gold_ans, cat, id_, src, pref):
        data["question"].append(q)
        data["options"].append(opts)
        data["gold_option"].append(gold_opt)
        data["gold_answer"].append(gold_ans)
        data["category"].append(cat)
        data["id"].append(id_)
        data["source"].append(src)
        data["preference"].append(pref)

    # Process TruthfulQA
    for i, (q, opt_obj) in tqdm(
        enumerate(zip(tqa_df["question"], tqa_df["mc1_targets"])),
        total=len(tqa_df), desc="Processing TruthfulQA"
    ):
        opts = list(opt_obj['choices'])
        answer = opts[0]
        shuffled_opts = random.sample(opts, len(opts))
        gold_opt = get_answer_letter_option(answer, shuffled_opts)
        append_row(i, q, shuffled_opts, gold_opt, answer, None, None, "truthfulqa/truthful_qa", _get_pref(TQA_PREFS, i))

    # Process CommonsenseQA
    for i, (id_, q, opts, key) in tqdm(
        enumerate(zip(cqa_df["id"], cqa_df["question"], cqa_df["choices"], cqa_df["answerKey"])),
        total=len(cqa_df), desc="Processing CommonsenseQA"
    ):
        opt_texts = list(opts['text'])
        gold_ans = get_answer_text(key, opt_texts)
        append_row(i, q, opt_texts, key, gold_ans, None, id_, "tau/commonsense_qa", _get_pref(CQA_PREFS, i))

    # Process MMLU
    for i, (q, opts, ans_idx, cat) in tqdm(
        enumerate(zip(mmlu_df["question"], mmlu_df["choices"], mmlu_df["answer"], mmlu_df["subject"])),
        total=len(mmlu_df), desc="Processing MMLU"
    ):
        answer_key = string.ascii_uppercase[ans_idx]
        answer = opts[ans_idx]
        append_row(i, q, opts, answer_key, answer, cat, None, "cais/mmlu", _get_pref(MMLU_PREFS, i))

    return pd.DataFrame(data)

def assign_preferences(df, prefs_dict):
    pref_ids = list(prefs_dict.keys())
    df = df.copy()
    df['preference_id'] = [pref_ids[i % len(pref_ids)] for i in range(len(df))]
    df['preference'] = df['preference_id'].map(prefs_dict)
    return df

def merge_df_responses(df_path: str, main_df_path:str):
    
    df = pd.read_csv(df_path)
    df_main = pd.read_csv(main_df_path)
    
    columns_to_rename = {
    'no_pref_ans': 'nopref_answer',
    'pref_ans': 'pref_answer',
    'no_pref_res': 'nopref_res',
    }
    
    model_filename = os.path.basename(df_path)
    model_name = model_filename.replace('.csv', '') 
    parts = df_path.split(os.sep)
    
    model_index = parts.index(model_filename)
    method_name = parts[model_index - 2]
    
    df.rename(columns={col: new_col for col, new_col in columns_to_rename.items() if col in df.columns}, inplace=True)
    df_main.rename(columns={col: new_col for col, new_col in columns_to_rename.items() if col in df_main.columns}, inplace=True)
    
    
    df = df.sort_values(by=['question', 'options']).reset_index(drop=True)
    df_main = df_main.sort_values(by=['question', 'options']).reset_index(drop=True)
    
    print(f"Length of df {len(df)}")
    print(f"Length of main_df {len(df_main)}")
    # df_relevant = df_relevant.drop_duplicates(subset='question')
    
    initial_nopref_acc = len(df_main[df_main['nopref_answer'] == df_main['gold_option']])/len(df_main)
    
    initial_pref_acc = len(df_main[df_main['pref_answer'] == df_main['gold_option']])/len(df_main)
    new_pref_acc = len(df[df['pref_answer'] == df['gold_option']])/len(df_main)
    
    df_merged = df
    df_merged['nopref_answer'] = df_main['nopref_answer']
    df_merged['nopref_res'] = df_main['nopref_res']
    
    print(f"Length of merged_df {len(df_merged)}")
    
    merged_nopref_acc = len(df_merged[df_merged['nopref_answer'] == df_merged['gold_option']])/len(df_merged)
    merged_pref_acc = len(df_merged[df_merged['pref_answer'] == df_merged['gold_option']])/len(df_merged)
    
    print(f"initial no pref accuracy is {initial_nopref_acc}")
    print(f"Merged df no pref accuracy is {merged_nopref_acc}")
    
    print(f"initial pref accuracy is {initial_pref_acc}")
    print(f"New df pref accuracy is {new_pref_acc}")
    
    print(f"Merged df pref accuracy is {merged_pref_acc}")
    
    save_path = df_path.replace('.csv', f'-full-new.csv')
    df_merged.to_csv(save_path, index=False)
    
    print(f"Saved merged file to: {save_path}")
    return save_path