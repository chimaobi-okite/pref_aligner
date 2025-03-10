# pref_aligner

## How TO Run
* First clone the project
* Load a python3.11.5 module
* Load Cuda
* create a virtual environment
* activate virtual environment
* pip install requirements.txt
* run run_mcq.sh to run evaluation on multichoice questions. Supported datasets for now are "truthfulqa/truthful_qa, tau/commonsense_qa, cais/mmlu". mistralai/Mistral-7B-Instruct-v0.3 has already been evaluated on this datasets and results stored in "results/mcq_results". You can start by evaluating other models probably higher mistral models or llama models etc to see how things. 
* run run_merge.sh with same dataset and model path used in above to aggregate results. folder_path here should be "mcq_results".