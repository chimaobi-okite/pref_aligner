#!/bin/bash

#SBATCH --mail-user=cokite@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=spgpu
#SBATCH --time=48:10:00
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-gpu=48GB
#SBATCH --account=mihalcea98 # cse692w25_class # mihalcea_owned1 # mihalcea98 # chaijy2  # mihalcea98 # mihalcea_owned1

# Load modules
module load python/3.11.5 cuda
source venv/bin/activate  # Ensure the virtual env path is correct

# Print GPU info for debugging
nvidia-smi

# Set variables for the run
PROMPT_METHOD="direct" # Options: "icl" or "cot" or "direct"
PREF_TYPE="relevant"       # Options: "relevant" or "irrelevant"
# MODEL_PATH="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
MODEL_PATH="gpt-4o-mini-2024-07-18"
# MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.3" # model you're running here

# Run job for the assigned chunk
CUDA_LAUNCH_BLOCKING=1 
python -m evaluations.full_main \
    --model_path=$MODEL_PATH \
    --pref_type=$PREF_TYPE \
    --prompt_method=$PROMPT_METHOD


# Data = 
#   -- mistralai/Mistral-7B-Instruct-v0.3
#   -- meta-llama/Llama-3.1-8B-Instruct
#   -- kaist-ai/janus-7b
#   -- meta-llama/Llama-3.3-70B-Instruct
#   -- mistralai/Mixtral-8x7B-Instruct-v0.1
#   -- update with any model you run