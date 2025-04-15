#!/bin/bash

#SBATCH --time=00-00:20:00
#SBATCH --account=chaijy2 # mihalcea_owned1

# set up job
module load python/3.11.5
source venv/bin/activate

# Set variables for the run
PROMPT_METHOD="cot"
FOLDER_PATH="mcq_results" # doesn't change for now
PREF_TYPE="relevant"    # Options: "relevant" or "irrelevant"
DATA_PATH="full" # data you're running here
MODEL_PATH="mistralai/Mixtral-8x7B-Instruct-v0.1" # model you're running here

# run job
python -m evaluations.merge_results \
    --model_path=$MODEL_PATH \
    --data_path=$DATA_PATH \
    --folder_path=$FOLDER_PATH \
    --pref_type=$PREF_TYPE \
    --prompt_method=$PROMPT_METHOD

    
    
    
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