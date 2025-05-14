#!/bin/bash

#SBATCH --mail-user=cokite@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=spgpu
#SBATCH --time=120:10:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-gpu=48GB
#SBATCH --account=chaijy2 # cse692w25_class # mihalcea_owned1 # mihalcea98 # chaijy2  # mihalcea98 # mihalcea_owned1

module load python/3.11.5 cuda
source venv/bin/activate  # Ensure the virtual env path is correct

nvidia-smi

# DF_PATH="results/mcq_results/relevant/direct/full/Mistral-7B-Instruct-v0.3-direct-full.csv"
# MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.3"

# DF_PATH="results/mcq_results/relevant/direct/full/Mixtral-8x7B-Instruct-v0.1-direct-full.csv"
# MODEL_PATH="mistralai/Mixtral-8x7B-Instruct-v0.1"

# DF_PATH="results/mcq_results/relevant/direct/full/Llama-3.1-8B-Instruct-direct-full.csv"
# MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

# DF_PATH="results/mcq_results/relevant/direct/full/janus-7b-direct-full.csv"
# MODEL_PATH="kaist-ai/janus-7b"

# DF_PATH="results/mcq_results/relevant/direct/full/Llama-3.3-70B-Instruct-Turbo-Free-direct-full_eval.csv"
# MODEL_PATH="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

DF_PATH="results/mcq_results/relevant/direct/full/gemma-2-9b-it-direct-full.csv"
MODEL_PATH="google/gemma-2-9b-it"

# DF_PATH="results/mcq_results/relevant/direct/full/Qwen3-8B-direct-full.csv"
# MODEL_PATH="Qwen/Qwen3-8B"



PREF_TYPE="relevant"       # Options: "relevant" or "irrelevant" or"irrelevant_set"

# Run job for the assigned chunk
CUDA_LAUNCH_BLOCKING=1 
python -m evaluations.aligner_main \
    --df_path=$DF_PATH \
    --model_path=$MODEL_PATH \
    --pref_type=$PREF_TYPE \


# Data = 
#   -- mistralai/Mistral-7B-Instruct-v0.3
#   -- meta-llama/Llama-3.1-8B-Instruct
#   -- kaist-ai/janus-7b
#   -- meta-llama/Llama-3.3-70B-Instruct
#   -- mistralai/Mixtral-8x7B-Instruct-v0.1
#   -- meta-llama/Llama-3.3-70B-Instruct-Turbo-Free

#   -- google/gemma-2-27b-it
#   -- google/gemma-2-9b-it
#   -- Qwen/Qwen3-8B
#   -- Qwen/Qwen3-8B-FP8
#   -- Qwen/Qwen3-8B
#   -- deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free
#   -- update with any model you run