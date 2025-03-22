#!/bin/bash

#SBATCH --mail-user=cokite@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=spgpu
#SBATCH --time=30:10:00
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-gpu=32GB
#SBATCH --account=chaijy2 # mihalcea_owned1 # mihalcea98 # chaijy2  # mihalcea98 # mihalcea_owned1
#SBATCH --array=0-29  # Run four parallel jobs (chunks 0, 1, 2, 3)

# Load modules
module load python/3.11.5 cuda
source venv/bin/activate  # Ensure the virtual env path is correct

# Print GPU info for debugging
nvidia-smi

# Set variables for the run
PREF_TYPE="relevant"       # Options: "relevant" or "irrelevant"
DATA_PATH="cais/mmlu" # data you're running here
MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.3" # model you're running here
CHUNK_SIZE=30 # your chunk size

# Run job for the assigned chunk
CUDA_LAUNCH_BLOCKING=1 
python -m evaluations.main \
    --chunk=${SLURM_ARRAY_TASK_ID} \
    --chunk_size=$CHUNK_SIZE \
    --model_path=$MODEL_PATH \
    --data_path=$DATA_PATH \
    --pref_type=$PREF_TYPE


# Data = 
#   -- tau/commonsense_qa
#   -- cais/mmlu
#   -- truthfulqa/truthful_qa
#   -- more data will be added here


# Data = 
#   -- mistralai/Mistral-7B-Instruct-v0.3
#   -- meta-llama/Llama-3.1-8B-Instruct
#   
#   -- update with any model you run