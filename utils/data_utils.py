import pandas as pd
from math import ceil
from sklearn.model_selection import train_test_split
from datasets import load_dataset
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