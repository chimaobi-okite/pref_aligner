#!/bin/bash

#SBATCH --time=00-00:20:00
#SBATCH --account=chaijy2 # mihalcea_owned1

# set up job
module load python/3.11.5
source venv/bin/activate

# Set variables for the run
FOLDER_PATH="mcq_results" # doesn't change for now
PREF_TYPE="relevant"       # Options: "relevant" or "irrelevant"
DATA_PATH="cais/mmlu" # data you're running here
MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.3" # model you're running here

# run job
python -m evaluations.merge_results \
    --model_path=$MODEL_PATH \
    --data_path=$DATA_PATH \
    --folder_path=$FOLDER_PATH \
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