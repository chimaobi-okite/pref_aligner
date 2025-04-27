#!/bin/bash

#SBATCH --time=48:10:00
#SBATCH --account=chaijy2
#SBATCH --mail-user=cokite@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=pref_eval_batch
#SBATCH --output=logs/pref_eval_%j.log

# Set up job
module load python/3.11.5
source venv/bin/activate

# List of data paths
DATA_PATHS=(
    # "results/mcq_results/irrelevant_set/direct/full/janus-7b.csv"
    # "results/mcq_results/irrelevant_set/direct/full/Llama-3.1-8B-Instruct.csv"

    # "results/mcq_results/irrelevant_set/direct/full/Mistral-7B-Instruct-v0.3.csv"
    # "results/mcq_results/irrelevant_set/direct/full/Mixtral-8x7B-Instruct-v0.1.csv"

    # "results/mcq_results/relevant/cot/full/Llama-3.3-70B-Instruct-Turbo-Free_7208_-0.0012486126526082275_0.9424237200482676.csv"
    # "results/mcq_results/relevant/icl/full/Llama-3.3-70B-Instruct-Turbo-Free.csv"
    # "results/mcq_results/relevant/direct/full/Llama-3.3-70B-Instruct-Turbo-Free_7208_0.007769145394006638_0.944060773480663.csv"
    "results/mcq_results/relevant/direct/full/gpt-4o-mini-2024-07-18_7208_0.04814095449500555_0.8851213637205855.csv"
)

# Loop through each data path
for DATA_PATH in "${DATA_PATHS[@]}"; do
    echo "Running pref_eval for $DATA_PATH"
    python -m evaluations.pref_eval --df_path="$DATA_PATH"
done

    
# results/mcq_results/relevant/direct/full/Mistral-7B-Instruct-v0.3-direct-full.csv
# results/mcq_results/relevant/direct/full/Mixtral-8x7B-Instruct-v0.1-direct-full.csv
# results/mcq_results/relevant/direct/full/janus-7b-direct-full.csv

# results/mcq_results/relevant/cot/full/janus-7b-cot-full.csv
# results/mcq_results/relevant/cot/full/Llama-3.1-8B-Instruct-cot-full.csv
# results/mcq_results/relevant/cot/full/Mistral-7B-Instruct-v0.3_7208_0.01220865704772467_0.8237736828445367.csv
# results/mcq_results/relevant/cot/full/Mixtral-8x7B-Instruct-v0.1_0.8159714476912782_0.013873473917868973_final.csv

# results/mcq_results/relevant/icl/full/janus-7b_7208_-0.06506659267480575_0.6539050535987749.csv
# results/mcq_results/relevant/icl/full/Llama-3.1-8B-Instruct-icl-full.csv
# results/mcq_results/relevant/icl/full/Mistral-7B-Instruct-v0.3-icl-full.csv
# results/mcq_results/relevant/icl/full/Mixtral-8x7B-Instruct-v0.1_0.8177570093457944_0.007630410654827946_final.csv

# DATA_PATHS=(
#     # "results/mcq_results/relevant/direct/full/Mixtral-8x7B-Instruct-v0.1-direct-full.csv"
#     # "results/mcq_results/relevant/direct/full/janus-7b-direct-full.csv"

#     # "results/mcq_results/relevant/cot/full/janus-7b-cot-full.csv"
#     # "results/mcq_results/relevant/cot/full/Mistral-7B-Instruct-v0.3_7208_0.01220865704772467_0.8237736828445367.csv"
#     "results/mcq_results/relevant/cot/full/Mixtral-8x7B-Instruct-v0.1_0.8159714476912782_0.013873473917868973_final.csv"
#     # uncomment up and down later
#     "results/mcq_results/relevant/icl/full/janus-7b_7208_-0.06506659267480575_0.6539050535987749.csv"
#     # "results/mcq_results/relevant/icl/full/Llama-3.1-8B-Instruct-icl-full.csv"
#     # "results/mcq_results/relevant/icl/full/Mistral-7B-Instruct-v0.3-icl-full.csv"
#     # uncoment down later
#     "results/mcq_results/relevant/icl/full/Mixtral-8x7B-Instruct-v0.1_0.8177570093457944_0.007630410654827946_final.csv"
# )