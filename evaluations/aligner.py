import ast
import os
import torch
from typing import Dict, List
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from openai import AzureOpenAI
from openai import OpenAI
from together import Together


from utils.janus_utils import apply_template_mistral_instruct, extract_after_inst
from utils.mcq_extractor_utils import extract_answer_letters_batch, extract_thoughts_and_response
from utils.mcq_utils import aligner_agent_message, calculate_accuracy, format_mcq_user_prompt, get_last_part
from utils.model_utils import load_janus_model, load_model, load_qwen_model
from utils.utils import get_messages

load_dotenv()

endpoint = os.getenv("AZUREOPENAI_ENDPOINT")  
deployment = "gpt-4o-mini"
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "REPLACE_WITH_YOUR_KEY_VALUE_HERE")
openai_api_key = os.getenv("OPENAI_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")

openai_client = OpenAI(api_key=openai_api_key)
together_client = Together(api_key=together_api_key)


client_models = ["meta-llama/Llama-3.3-70B-Instruct-Turbo",
                 "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                 "gpt-4o-mini-2024-07-18", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"]


def client_generate(client, model, messages:List):
    response = client.chat.completions.create(
            model=model, # "gpt-4-32k", # model = "deployment_name".
            # model = deployment,
            messages=messages,
            temperature=0,
            seed=42,
        )
    content = response.choices[0].message.content
    if not content:
        print(messages)
        print(response.choices[0].message.content)
        content = "None"
    return content
    


def run_client_aligner_generation(user_messages, batch_options_list,batch_responses, client, model,):
        
    results = []
    input_len = len(user_messages)
    for messages in user_messages:
        content = client_generate(client, model, messages=messages)
        results.append(content)
    list_of_references = batch_options_list[:input_len]
    list_of_references = [ast.literal_eval(refs) for refs in list_of_references]
    
    results = [extract_thoughts_and_response(response) for response in results]
    responses = []
    thoughts = []
    for fallback_response, parsed in zip(batch_responses, results):
        if parsed and 'response' in parsed:
            responses.append(parsed['response'])
            thoughts.append(parsed.get('thought', None))
        else:
            responses.append(fallback_response)
            thoughts.append(None)
    
    predictions, invalids = extract_answer_letters_batch(responses, list_of_references)
    return responses, predictions, invalids, thoughts

def run_aligner_generation(user_messages, batch_options_list, batch_responses, generator,):
    results = generator(user_messages, max_new_tokens=1000,)
    input_len = len(user_messages)
    results = [results[i][0]["generated_text"][-1]['content'] for i in range(0, input_len)]
    list_of_references = batch_options_list[:input_len]
    list_of_references = [ast.literal_eval(refs) for refs in list_of_references]
    
    results = [extract_thoughts_and_response(response) for response in results]
    responses = []
    thoughts = []
    for fallback_response, parsed in zip(batch_responses, results):
        if parsed and 'response' in parsed:
            responses.append(parsed['response'])
            thoughts.append(parsed.get('thought', None))
        else:
            responses.append(fallback_response)
            thoughts.append(None)
    
    # print(responses)
    # print(thoughts)
    
    predictions, invalids = extract_answer_letters_batch(responses, list_of_references,)
    return responses, predictions, invalids, thoughts

def run_janus_aligner_generation(user_messages, batch_options_list, batch_responses, janus, janus_tokenizer,):
    input_strs = [apply_template_mistral_instruct(user_message) for 
                   user_message in user_messages]
    input_len = len(input_strs)
    inputs = janus_tokenizer(input_strs, return_tensors="pt", padding=True, truncation=True).to(janus.device)
    output_ids = janus.generate(inputs['input_ids'], max_new_tokens=1024, temperature=0.0,do_sample=False,
                                pad_token_id=janus_tokenizer.eos_token_id)
    results = janus_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    results = [extract_after_inst(res) for res in results]
    list_of_references = batch_options_list[:input_len]
    list_of_references = [ast.literal_eval(refs) for refs in list_of_references]
    
    results = [extract_thoughts_and_response(response) for response in results]
    responses = []
    thoughts = []
    for fallback_response, parsed in zip(batch_responses, results):
        if parsed and 'response' in parsed:
            responses.append(parsed['response'])
            thoughts.append(parsed.get('thought', None))
        else:
            responses.append(fallback_response)
            thoughts.append(None)
    predictions, invalids = extract_answer_letters_batch(responses, list_of_references)
    return responses, predictions, invalids, thoughts


def run_aligner(df_path:str, model_path, pref_type):
    
    print(f"Running aligner model model {model_path}, on pref type {pref_type}")
    
    BATCH_SIZE=32
    is_janus = False
    is_qwen = False
    is_client_model = False
   
    df = pd.read_csv(df_path)

    columns_to_rename = {
        'no_pref_ans': 'nopref_answer',
        'pref_ans': 'pref_answer',
        'no_pref_res': 'nopref_res',
    }
    
    df.rename(columns={col: new_col for col, new_col in columns_to_rename.items() if col in df.columns}, inplace=True)
    # print(df[['question', 'nopref_res', 'options', 'preference']].isnull().sum())
    # df = df[:60]
    
    
    
    if "janus" in model_path:
        model, tokenizer = load_janus_model(model_path=model_path)
        is_janus = True
    elif "Qwen3" in model_path:
        print("running Qwen")
        model, tokenizer = load_qwen_model(model_path=model_path)
        is_qwen = True
    elif model_path in client_models:
        is_client_model = True
        if "gpt" in model_path:
            client = openai_client
            print("gpt")
        else:
            client = together_client
    else:
        generator = load_model(model_path)
        
        
    aligner_res = []
    aligner_ans = []
    
    aligner_invalids = []
    
    all_thoughts = []
    
    questions = df["question"].tolist()
    nopref_responses = df["nopref_res"].tolist()
    options_list = df["options"].tolist()
    preferences = df["preference"].tolist()
    
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Processing Batches"):
        batch_questions = questions[i:i+BATCH_SIZE]
        batch_responses =  nopref_responses[i:i+BATCH_SIZE]
        batch_options_list = options_list[i:i+BATCH_SIZE]
        batch_preferences = preferences[i: i+BATCH_SIZE]
        
        # print(f"====running batch ==== {i}\n query {raw_user_messages}")
        
    
        user_queries = [format_mcq_user_prompt(question, ast.literal_eval(options)) 
                            for (question, options) in zip(batch_questions, batch_options_list)]
        raw_user_messages = [aligner_agent_message(query, response, preference) 
                             for (query, response, preference) in 
                             zip(user_queries, batch_responses, batch_preferences)]
        user_messages = [get_messages(user_message=msg) for msg in raw_user_messages]
        
        if is_janus:
            responses, predictions, invalids, thoughts = run_janus_aligner_generation(
                    user_messages, batch_options_list,batch_responses,
                    janus=model, janus_tokenizer=tokenizer
                )
                
        elif is_client_model:
            responses, predictions, invalids, thoughts = run_client_aligner_generation(
                    user_messages, batch_options_list, batch_responses,
                    client=client, model=model_path)
            
        else:
            responses, predictions, invalids, thoughts = run_aligner_generation(
                    user_messages, batch_options_list, batch_responses, generator,
                )
        
        
        aligner_res.extend(responses)
        aligner_ans.extend(predictions)
        aligner_invalids.extend(invalids)
        all_thoughts.extend(thoughts)  
                  
        torch.cuda.empty_cache()
        
    df["aligner_res"] = aligner_res
    df["aligner_answer"] = aligner_ans
    df["aligner_invalid"] = aligner_invalids
    df["aligner_thoughts"] = all_thoughts
    
    final_df = post_process_mcq_results(df, model_path, pref_type)
    
def post_process_mcq_results(df: pd.DataFrame, model_path, pref_type):

    aligner_acc = calculate_accuracy(df, f'aligner_answer')
    pref_acc = calculate_accuracy(df, f'pref_answer')
    no_pref_acc = calculate_accuracy(df, f'nopref_answer')
        
    print(f'Pref_answer_accuracy is: {pref_acc}')
    print(f'No_Pref_answer_accuracy is: {no_pref_acc}')
    print(f'aligner_answer_accuracy is: {aligner_acc}')
    
        
    model_name =get_last_part(model_path)
   
    if no_pref_acc:
        acc_diff = no_pref_acc - pref_acc
        robustness = get_aligner_robustness(df)
        print(f"Acc diff {acc_diff} and robustness is {robustness}")
    
    save_csv(df = df, pref_type = pref_type,model_name = model_name, )
    return df

def get_aligner_robustness(df):
    df = df.copy()
    df = df[df['nopref_answer'] == df['gold_option']]
    accuracy_result = calculate_accuracy(df, "aligner_answer")
    return accuracy_result

def save_csv(df:pd.DataFrame, pref_type: str, model_name:str):
    save_path = f"results/mcq_results/{pref_type}/aligner/full"
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(f"{save_path}/{model_name}.csv", index=False)
    
    

