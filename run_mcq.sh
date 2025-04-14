#!/bin/bash

#SBATCH --mail-user=cokite@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=spgpu
<<<<<<< HEAD
#SBATCH --time=3:10:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-gpu=32GB
#SBATCH --account=cse692w25_class # cse692w25_class # mihalcea_owned1 # mihalcea98 # chaijy2  # mihalcea98 # mihalcea_owned1
# SBATCH --array=0-29%6  # Run four parallel jobs (chunks 0, 1, 2, 3)
=======
#SBATCH --time=30:10:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-gpu=32GB
#SBATCH --account=mihalcea_owned1 # cse692w25_class # mihalcea_owned1 # mihalcea98 # chaijy2  # mihalcea98 # mihalcea_owned1
#SBATCH --array=0-29%6  # Run four parallel jobs (chunks 0, 1, 2, 3)
>>>>>>> 68864568f6f010486372b2e007ca5bec8e14a552

# Load modules
module load python/3.11.5 cuda
source venv/bin/activate  # Ensure the virtual env path is correct

# Print GPU info for debugging
nvidia-smi

# Set variables for the run
<<<<<<< HEAD
PROMPT_METHOD="direct" # Options: "icl" or "cot" or "direct"
PREF_TYPE="relevant"       # Options: "relevant" or "irrelevant"
DATA_PATH="truthfulqa/truthful_qa" # data you're running here
MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.3" # model you're running here
# MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
CHUNK_SIZE=18 # your chunk size
SLURM_ARRAY_TASK_ID=0
=======
PROMPT_METHOD="icl" # Options: "icl" or "cot" or "direct"
PREF_TYPE="relevant"       # Options: "relevant" or "irrelevant"
DATA_PATH="cais/mmlu" # data you're running here
# MODEL_PATH="mistralai/Mixtral-8x7B-Instruct-v0.1" # model you're running here
MODEL_PATH="kaist-ai/janus-7b"
# MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
CHUNK_SIZE=30 # your chunk size

>>>>>>> 68864568f6f010486372b2e007ca5bec8e14a552
# Run job for the assigned chunk
CUDA_LAUNCH_BLOCKING=1 
python -m evaluations.main \
    --chunk=${SLURM_ARRAY_TASK_ID} \
    --chunk_size=$CHUNK_SIZE \
    --model_path=$MODEL_PATH \
    --data_path=$DATA_PATH \
    --pref_type=$PREF_TYPE \
    --prompt_method=$PROMPT_METHOD


# ${SLURM_ARRAY_TASK_ID} \
# Data = 
#   -- tau/commonsense_qa
#   -- cais/mmlu
#   -- truthfulqa/truthful_qa
#   -- more data will be added here


# Data = 
#   -- mistralai/Mistral-7B-Instruct-v0.3
#   -- meta-llama/Llama-3.1-8B-Instruct
#   -- kaist-ai/janus-7b
#   -- meta-llama/Llama-3.3-70B-Instruct
#   -- mistralai/Mixtral-8x7B-Instruct-v0.1
#   -- update with any model you run