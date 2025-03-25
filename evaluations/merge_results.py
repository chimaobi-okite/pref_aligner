import argparse
import os
import glob
import re
import pandas as pd
from typing import Dict

from enums import PrefType
from utils.mcq_utils import calculate_accuracy, calculate_math_accuracy, get_last_part

def aggregate_mcq_results(model_path: str, data_path: str, folder_path:str, pref_type: str):
    """Aggregates all chunked CSV results, deletes chunk files, and calculates accuracy."""
    

    # output_dir=f"results/{folder_path}"
    model_name = get_last_part(model_path)
    data_name = get_last_part(data_path)
    output_dir = f"results/{folder_path}/{pref_type}/{data_name}"
    
    chunk_files_pattern = f"{output_dir}/{model_name}_*.csv"
    all_files = glob.glob(chunk_files_pattern)
    

    chunk_files = []
    for f in all_files:
        match = re.search(rf"{re.escape(model_name)}_(\d+)\.csv", f)
        if match and 0 <= int(match.group(1)) <= 50:
            chunk_files.append(f)
    # # Filter for chunks between 0 and 20
    # chunk_files = [
    #     f for f in chunk_files if (match := re.search(rf"{re.escape(model_name)}_(\d+)\.csv", f)) 
    #     and 0 <= int(match.group(1)) <= 20
    # ]
        
    if not chunk_files:
        print("No chunk files found for aggregation.")
        return None

    print(f"Found {len(chunk_files)} chunk files. Aggregating results...")
    
    # Merge all CSV files
    merged_df = pd.concat((pd.read_csv(f) for f in chunk_files), ignore_index=True)

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

    # Display the accuracy results
    for method, accuracy in accuracy_results.items():
        print(f"{method}: {accuracy:.2%}")
        
        
    print(f"lenght of final df is {len(merged_df)}")

    # Save final aggregated CSV
    final_output_path = f"{output_dir}/{model_name}_final.csv"
    merged_df.to_csv(final_output_path, index=False)
    
    print(f"Aggregated results saved at: {final_output_path}")

    return merged_df, accuracy_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Main Job")
    parser.add_argument('--data_path', type=str, required=True, help="mcq_dataset_path")
    parser.add_argument('--model_path', type=str, required=True, help="mcq_model_path")
    parser.add_argument('--folder_path', type=str, required=True, help="sub folder under results to store data")
    parser.add_argument('--pref_type', type=str, choices=[e.value for e in PrefType], required=True, help="whether relevant or irrevant")
    args = parser.parse_args()
    merged_df, accuracy_results = aggregate_mcq_results(args.model_path, args.data_path, args.folder_path, args.pref_type)