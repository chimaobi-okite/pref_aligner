#!/bin/bash

#SBATCH --mail-user=cokite@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=spgpu
#SBATCH --time=15:10:00
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-gpu=32GB
#SBATCH --account=mihalcea98 # mihalcea98 # chaijy2  # mihalcea98 # mihalcea_owned1
#SBATCH --array=0-9  # Run four parallel jobs (chunks 0, 1, 2, 3)

# Load modules
module load python/3.11.5 cuda
source venv/bin/activate  # Ensure the virtual env path is correct

# Print GPU info for debugging
nvidia-smi

# Run job for the assigned chunk
# CUDA_LAUNCH_BLOCKING=1 python -m evaluations.main \
#     --chunk=${SLURM_ARRAY_TASK_ID} \
#     --chunk_size=4 \
#     --model_path=mistralai/Mistral-7B-Instruct-v0.3 \
#     --data_path=truthfulqa/truthful_qa

# CUDA_LAUNCH_BLOCKING=1 python -m evaluations.main \
#     --chunk=${SLURM_ARRAY_TASK_ID} \
#     --chunk_size=4 \
#     --model_path=mistralai/Mistral-7B-Instruct-v0.3 \
#     --data_path=tau/commonsense_qa

# CUDA_LAUNCH_BLOCKING=1 python -m evaluations.main \
#     --chunk=4  \
#     --chunk_size=5 \
#     --model_path=mistralai/Mistral-7B-Instruct-v0.3 \
#     --data_path=cais/mmlu

CUDA_LAUNCH_BLOCKING=1 python -m evaluations.main \
    --chunk=${SLURM_ARRAY_TASK_ID}  \
    --chunk_size=10 \
    --model_path=mistralai/Mistral-7B-Instruct-v0.3 \
    --data_path=openai/gsm8k
# meta-llama/Meta-Llama-3.1-8B-Instruct