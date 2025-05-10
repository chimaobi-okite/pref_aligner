import os
import re
import ast
import string
import pandas as pd
from openai import AzureOpenAI
from openai import OpenAI
from together import Together
from dotenv import load_dotenv

load_dotenv()

endpoint = os.getenv("AZUREOPENAI_ENDPOINT")  
deployment = "gpt-4o-mini"
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "REPLACE_WITH_YOUR_KEY_VALUE_HERE")
openai_api_key = os.getenv("OPENAI_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")


# client = OpenAI(api_key=openai_api_key)

client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key,  
    api_version="2025-01-01-preview",
)

def generate_preference_judge_system_prompt(user_prompt: str, preference: str,
                                             response_without_preference: str,
                                             response_with_preference: str) -> str:
    return f"""
You are a careful and neutral evaluator whose job is to assess how well a language model adjusts its response based on a specific user preference.

You will be shown the following:
- A user prompt
- A user preference
- The model's original response *without* being given the preference
- The model's response *after* being conditioned on the preference

Your task is to rate how well the preference is followed in the second response using a **Likert scale from 1 (Strongly Disagree) to 5 (Strongly Agree)**.

You should **only** consider the degree to which the response follows the given preference, regardless of factual correctness or answer quality.

---

**User Prompt:**  
{user_prompt}

**User Preference:**  
{preference}

**Model Response Without Preference:**  
{response_without_preference}

**Model Response With Preference:**  
{response_with_preference}

---

### Likert Scoring Guide:
- **1 – Strongly Disagree:** The response does not follow the preference at all.
- **2 – Disagree:** The response acknowledges the preference implicitly or explicitly, but does not seem to follow it meaningfully.
- **3 – Neutral:** The response reflects a moderate or partial attempt to follow the preference.
- **4 – Agree:** The response shows clear effort to follow the preference and mostly succeeds.
- **5 – Strongly Agree:** The response fully aligns with the preference in a coherent and meaningful way.

Output your final decision in **this exact XML format**:
```xml
<explanation>[One short sentence explaining your score]</explanation>
<answer>[1-5]</answer>
"""



def get_pref_evaluation(user_prompt: str, 
                        preference: str,
                        response_without_preference: str,
                        response_with_preference: str):
  
  system_prompt = generate_preference_judge_system_prompt(user_prompt, preference,
                                                           response_without_preference,
                                                           response_with_preference)
  messages=[
        {"role": "system", "content": system_prompt},
        ]
  response = client.chat.completions.create(
        model=deployment, # "gpt-4-32k", # model = "deployment_name".
        messages=messages,
        temperature=0,
        seed=42,
    )
  prompt_tokens = response.usage.prompt_tokens
  completion_tokens = response.usage.completion_tokens
  return response.choices[0].message.content, prompt_tokens, completion_tokens

def extract_judgment_from_xml(xml_string: str):
    explanation_match = re.search(r"<explanation>(.*?)</explanation>", xml_string, re.DOTALL)
    answer_match = re.search(r"<answer>(\d+)</answer>", xml_string)

    explanation = explanation_match.group(1).strip() if explanation_match else None
    answer = int(answer_match.group(1)) if answer_match else None

    return explanation, answer

def estimate_cost(prompt_tokens, completion_tokens, input_price_per_1m=0.15, output_price_per_1m=0.6):
    input_cost = (prompt_tokens / 1000000) * input_price_per_1m
    output_cost = (completion_tokens / 1000000) * output_price_per_1m
    return round(input_cost + output_cost, 6)


def preprocess_df_for_pref_eval(df_path:str, 
                                main_path:str = "data/robuset_main.csv") -> pd.DataFrame:
    df = pd.read_csv(df_path)
    columns_to_rename = {
    'no_pref_ans': 'nopref_answer',
    'pref_ans': 'pref_answer',
    'no_pref_res': 'nopref_res',
    }

    # rename columns if necessary 
    df.rename(columns={col: new_col for col, new_col in columns_to_rename.items() if col in df.columns}, inplace=True)
    
    robo_df = pd.read_csv(main_path)
    df['dup_id'] = df.groupby('question').cumcount()
    robo_df['dup_id'] = robo_df.groupby('question').cumcount()
    merged = df.merge(robo_df[['question', 'dup_id', 'options']], on=['question', 'dup_id'], how='inner')
    merged['options_x'] = None
    merged['options_x'] = merged['options_y']
    merged = merged.drop(columns=['options_y', 'dup_id'])
    merged = merged.rename(columns={'options_x': 'options'})
    return merged

def rename_path(old_path):
    folder = os.path.dirname(old_path)
    filename = os.path.basename(old_path)
    model_name = re.sub(r'(_\d+.*)', '', filename)
    if not 'csv' in model_name:
        new_filename = model_name + '.csv'
    else:
        new_filename = model_name
    new_path = os.path.join(folder, new_filename)
    os.rename(old_path, new_path)
    print(f"Renamed:\nFrom: {old_path}\nTo:   {new_path}")
    return new_path
    

def preprocess_aligner_df(df):
    main_df = df.copy()
    
    cols_to_drop = ['pref_correct','pref_eval_explanation','pref_eval_rating', 'pref_eval_prompt_tokens',
                    'pref_eval_completion_tokens', 'pref_eval_cost', 'pref_correct',
                    'nopref_correct', 'followed_pref', 'is_robust',]
    columns_to_rename = columns_to_rename = {
        'pref_answer': 'old_pref_answer',
        'pref_res': 'old_pref_res',
        'pref_correct': 'old_pref_correct'
        }
        
    main_df.rename(columns={col: new_col for col, new_col in columns_to_rename.items() if col in main_df.columns}, inplace=True)
    main_df = main_df.drop(columns=[col for col in cols_to_drop if col in main_df.columns])

    columns_to_rename = columns_to_rename = {
        'aligner_answer': 'pref_answer',
        'aligner_res': 'pref_res',
        }
    main_df.rename(columns={col: new_col for col, new_col in columns_to_rename.items() if col in main_df.columns}, inplace=True)
    print(main_df.columns)
    return main_df


def get_full_df(df_path: str):
    relevant_path = None
    
    if "Llama-3.3-70B-Instruct-Turbo-Free" in df_path: #and "icl" in df_path:
        df_path = rename_path(df_path)
        relevant_path = f"results/mcq_results/relevant/direct/full/Llama-3.3-70B-Instruct-Turbo-Free.csv"
    
    model_filename = os.path.basename(df_path)
    model_name = model_filename.replace('.csv', '') 
    parts = df_path.split(os.sep)
    
    model_index = parts.index(model_filename)
    method_name = parts[model_index - 2]
    
    if not relevant_path:
        relevant_path = f"results/mcq_results/relevant/direct/full/{model_name}-direct-full.csv"
    
    df_irrel = pd.read_csv(df_path)
    df_relevant = pd.read_csv(relevant_path)
    
    columns_to_rename = {
    'no_pref_ans': 'nopref_answer',
    'pref_ans': 'pref_answer',
    'no_pref_res': 'nopref_res',
    }
    
    df_irrel.rename(columns={col: new_col for col, new_col in columns_to_rename.items() if col in df_irrel.columns}, inplace=True)
    df_relevant.rename(columns={col: new_col for col, new_col in columns_to_rename.items() if col in df_relevant.columns}, inplace=True)
    
    # print("df_irrel columns:", df_irrel.columns)
    # print("df_relevant columns:", df_relevant.columns)
    
    cols_to_drop = ['nopref_res', 'nopref_answer']
    df_irrel = df_irrel.drop(columns=[col for col in cols_to_drop if col in df_irrel.columns])
    

    df_relevant = df_relevant.drop_duplicates(subset='question')
    
    initial_nopref_acc = len(df_relevant[df_relevant['nopref_answer'] == df_relevant['gold_option']])/len(df_relevant)
    
    initial_pref_acc = len(df_relevant[df_relevant['pref_answer'] == df_relevant['gold_option']])/len(df_relevant)
    new_pref_acc = len(df_relevant[df_irrel['pref_answer'] == df_irrel['gold_option']])/len(df_relevant)
    
    df_merged = df_irrel.merge(
        df_relevant[['question', 'nopref_res', 'nopref_answer']],
        on='question',
        how='left'
    )
    
    # print("df_merged columns:", df_merged.columns)

    
    merged_nopref_acc = len(df_merged[df_merged['nopref_answer'] == df_merged['gold_option']])/len(df_merged)
    merged_pref_acc = len(df_merged[df_merged['pref_answer'] == df_merged['gold_option']])/len(df_merged)
    
    print(f"initial no pref accuracy is {initial_nopref_acc}")
    print(f"Merged df no pref accuracy is {merged_nopref_acc}")
    
    print(f"initial pref accuracy is {initial_pref_acc}")
    print(f"New df pref accuracy is {new_pref_acc}")
    
    print(f"Merged df pref accuracy is {merged_pref_acc}")
    
    save_path = df_path.replace('.csv', f'-{method_name}-full.csv')
    df_merged.to_csv(save_path, index=False)
    
    print(f"Saved merged file to: {save_path}")
    return save_path
    