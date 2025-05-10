#!/bin/bash

#SBATCH --mail-user=cokite@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=spgpu
#SBATCH --time=120:10:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-gpu=48GB
#SBATCH --account=chaijy2 # cse692w25_class # mihalcea_owned1 # mihalcea98 # chaijy2  # mihalcea98 # mihalcea_owned1
# SBATCH --array=0-4 

# Load modules
module load python/3.11.5 cuda
source venv/bin/activate  # Ensure the virtual env path is correct

# Print GPU info for debugging
nvidia-smi

# Set variables for the run
# PROMPT_METHOD="direct" # Options: "icl" or "cot" or "direct" or "self_critic"
PROMPT_METHOD="direct"
ENABLE_THINKING=1
PREF_TYPE="irrelevant_set"       # Options: "relevant" or "irrelevant" or"irrelevant_set"
# PREF_TYPE="relevant" 
# MODEL_PATH="google/gemma-2-9b-it"
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
# MODEL_PATH="Qwen/Qwen3-8B"
# MODEL_PATH="Qwen/Qwen3-32B"

CHUNK_SIZE=5

# Run job for the assigned chunk
CUDA_LAUNCH_BLOCKING=1 
python -m evaluations.full_main \
    --model_path=$MODEL_PATH \
    --pref_type=$PREF_TYPE \
    --prompt_method=$PROMPT_METHOD \
    --enable_thinking=$ENABLE_THINKING \
    --chunk=1 \
    --chunk_size=$CHUNK_SIZE \


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