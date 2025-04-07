import argparse
import os
import glob
import re
import numpy as np
import pandas as pd
from typing import Dict

from enums import PrefType, PromptMethod
from utils.data_utils import load_data
from utils.mcq_utils import calculate_accuracy, calculate_math_accuracy, get_last_part


def get_agg_robustness(df, pref_columns):
    df = df.copy()
    df = df[df['profile_0_answer'] == df['gold_option']]
    accuracy_results = [calculate_accuracy(df, col) for col in pref_columns]
    return np.mean(accuracy_results[1:])  # Skips profile_0

def aggregate_mcq_results(model_path: str, data_path: str, folder_path:str, pref_type: str, prompt_method: str):
    """Aggregates all chunked CSV results, deletes chunk files, and calculates accuracy."""
    

    # output_dir=f"results/{folder_path}"
    model_name = get_last_part(model_path)
    data_name = get_last_part(data_path)
    if prompt_method:
        output_dir = f"results/{folder_path}/{pref_type}/{prompt_method}/{data_name}"
    else:
        output_dir = f"results/{folder_path}/{pref_type}/{data_name}"
    
    chunk_files_pattern = f"{output_dir}/{model_name}_*.csv"
    all_files = glob.glob(chunk_files_pattern)

    print(f"Searching {chunk_files_pattern}")

    chunk_files = []
    for f in all_files:
        match = re.search(rf"{re.escape(model_name)}_(\d+)\.csv", f)
        if match and 0 <= int(match.group(1)) <= 50:
            chunk_files.append(f)
        
    if not chunk_files:
        print("No chunk files found for aggregation.")
        return None

    print(f"Found {len(chunk_files)} chunk files. Aggregating results...")
    
    # Merge all CSV files
    merged_df = pd.concat((pd.read_csv(f) for f in chunk_files), ignore_index=True)
    
    if data_path == "truthfulqa/truthful_qa":
        data = load_data(data_path, "multiple_choice")
    else:
        data = load_data(data_path)
    if len(merged_df) != len(data):
        raise ValueError(f"Length mismatch: merged_df has {len(merged_df)} rows but data has {len(data)} entries.")


    # Delete chunk files
    for file in chunk_files:
        os.remove(file)
        print(f"Deleted: {file}")

    if "MATH" in data_path:
        profile_columns = [col for col in merged_df.columns if col.startswith("profile_") and col.endswith("_is_same")]
        accuracy_results = {
            f'{col}_accuracy': calculate_math_accuracy(merged_df, col) for col in profile_columns
        }
    else:
        profile_columns = [col for col in merged_df.columns if col.startswith("profile_") and col.endswith("_answer")]
        accuracy_results = {
            f'{col}_accuracy': calculate_accuracy(merged_df, col) for col in profile_columns
        }
        robustness = get_agg_robustness(merged_df, profile_columns)

    # Display the accuracy results
    for method, accuracy in accuracy_results.items():
        print(f"{method}: {accuracy:.2%}")
        
        
    print(f"lenght of final df is {len(merged_df)}")
    print(f"Robustness is {robustness}")
    
    

    # Save final aggregated CSV
    final_output_path = f"{output_dir}/{model_name}_{robustness}_final.csv"
    merged_df.to_csv(final_output_path, index=False)
    
    print(f"Aggregated results saved at: {final_output_path}")

    return merged_df, accuracy_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Main Job")
    parser.add_argument('--data_path', type=str, required=True, help="mcq_dataset_path")
    parser.add_argument('--model_path', type=str, required=True, help="mcq_model_path")
    parser.add_argument('--folder_path', type=str, required=True, help="sub folder under results to store data")
    parser.add_argument('--pref_type', type=str, choices=[e.value for e in PrefType], required=True, help="whether relevant or irrevant")
    parser.add_argument('--prompt_method', type=str, choices=[e.value for e in PromptMethod], required=True, help="whether icl, cot or direct")
    args = parser.parse_args()
    merged_df, accuracy_results = aggregate_mcq_results(args.model_path, args.data_path, args.folder_path, args.pref_type, args.prompt_method)
