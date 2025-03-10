import argparse

from evaluations.math_task import gms8k_task
from evaluations.mcq_task import common_sense_qa_task, mmlu_task, truthful_qa_task



def main(chunk:int, chunk_size:int, model_path: str, data_path:str):
    if "truthfulqa" in data_path:
        truthful_qa_task(chunk, chunk_size, model_path,)
    if "commonsense_qa" in data_path:
        common_sense_qa_task(chunk, chunk_size, model_path)
    if "mmlu" in data_path:
        mmlu_task(chunk, chunk_size, model_path,)
    if "openai/gsm8k" in data_path:
        gms8k_task(chunk, chunk_size, model_path,)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Main Job")
    parser.add_argument('--chunk', type=int, required=True, help="Chunk number to process (0 to 4)")
    parser.add_argument('--chunk_size', type=int, required=True, help="total number of chunks")
    parser.add_argument('--data_path', type=str, required=True, help="mcq_dataset_path")
    parser.add_argument('--model_path', type=str, required=True, help="mcq_model_path")
    args = parser.parse_args()
    main(args.chunk, args.chunk_size, args.model_path, args.data_path)