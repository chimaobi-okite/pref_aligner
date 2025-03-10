import argparse
import os
import glob
import pandas as pd
from typing import Dict

from utils.mcq_utils import calculate_accuracy, get_last_part

def aggregate_mcq_results(model_path: str, data_path: str, folder_path:str):
    """Aggregates all chunked CSV results, deletes chunk files, and calculates accuracy."""
    
    # mcq_results
    output_dir=f"results/{folder_path}"
    model_name = get_last_part(model_path)
    data_name = get_last_part(data_path)
    
    chunk_files_pattern = f"{output_dir}/{model_name}_{data_name}_*.csv"
    chunk_files = glob.glob(chunk_files_pattern)
    
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

    profile_columns = [col for col in merged_df.columns if col.startswith("profile_") and col.endswith("_answer")]
    accuracy_results = {
        f'{col}_accuracy': calculate_accuracy(merged_df, col) for col in profile_columns
    }

    # Display the accuracy results
    for method, accuracy in accuracy_results.items():
        print(f"{method}: {accuracy:.2%}")
        
        
    print(f"lenght of final df is {len(merged_df)}")

    # Save final aggregated CSV
    final_output_path = f"{output_dir}/{model_name}_{data_name}_final.csv"
    merged_df.to_csv(final_output_path, index=False)
    
    print(f"Aggregated results saved at: {final_output_path}")

    return merged_df, accuracy_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Main Job")
    parser.add_argument('--data_path', type=str, required=True, help="mcq_dataset_path")
    parser.add_argument('--model_path', type=str, required=True, help="mcq_model_path")
    parser.add_argument('--folder_path', type=str, required=True, help="sub folder under results to store data")
    args = parser.parse_args()
    merged_df, accuracy_results = aggregate_mcq_results(args.model_path, args.data_path, args.folder_path)