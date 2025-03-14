#!/bin/bash

#SBATCH --time=00-00:20:00
#SBATCH --account=chaijy2 # mihalcea_owned1

# set up job
module load python/3.11.5
source venv/bin/activate


# run job
python -m evaluations.merge_results \
    --model_path=mistralai/Mistral-7B-Instruct-v0.3 \
    --data_path=truthfulqa/truthful_qa \
    --folder_path=mcq_results

# python -m evaluations.merge_results \
#     --model_path=mistralai/Mistral-7B-Instruct-v0.3 \
#     --data_path=tau/commonsense_qa \
#     --folder_path=mcq_results

# python -m evaluations.merge_results \
#     --model_path=mistralai/Mistral-7B-Instruct-v0.3 \
#     --data_path=cais/mmlu \
#     --folder_path=mcq_results

# python -m evaluations.merge_results \
#     --model_path=mistralai/Mistral-7B-Instruct-v0.3 \
#     --data_path=openai/gsm8k \
#     --folder_path=math_results

# python -m evaluations.merge_results \
#     --model_path=mistralai/Mistral-7B-Instruct-v0.3 \
#     --data_path=xDAN2099/lighteval-MATH \
#     --folder_path=math_results
