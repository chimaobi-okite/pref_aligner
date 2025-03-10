import pandas as pd
from math import ceil
from datasets import load_dataset
SEED = 42

mmlu_subjects = ["college_physics", "college_mathematics","college_biology", "college_chemistry",
                 "business_ethics", "conceptual_physics", "anatomy"]

def load_data(path:str, split:str = None, sample_size: int = None):
    if "mmlu" in path:
        df = load_mmlu(path)
        if sample_size:
            df = df.sample(sample_size, random_state=SEED)
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
    df_list = [] 
    for sub in mmlu_subjects:
        ds = load_dataset(path, sub, split = "test")
        df_sub = ds.to_pandas()
        df_list.append(df_sub)

    df = pd.concat(df_list, ignore_index=True)

    print("Total samples after filtering:", len(df))
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