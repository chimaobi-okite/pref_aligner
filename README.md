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


## Models to Test
1. Mistral (7B, Mixtral 8x7B, Mixtral 8x22B)
2. LLama (8B, 13B, 70B)
3. Janus 7B [Aligning to Thousands of Preferences via System Message Generalization](https://arxiv.org/abs/2405.17977)

## Datasets to Test
1. MMLU
2. TruthfulQA
3. CommonsenseQA

Tasks (All task above have a timeline of Two Weeks. Person running the models should be done by next week so the analysis can follow):
1. Test Mixtral 8x7B, Mixtral 8x22B on all datasets above
2. Test 8B, 13B, 70B on all datasets above
3. Test Janus 7B on all the datasets above
4. Analyse Results -
   * Do both quantitative (see models accuracies and plot results) and
   * qualitative analysis. That is probe individual failure cases and make conclusions.
   * Do inter model analysis - any instances that continously fail across all models?
