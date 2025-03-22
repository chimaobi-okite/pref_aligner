import argparse
import json
import os
from typing import List
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from openai import AzureOpenAI
from openai import OpenAI
from together import Together

from dotenv import load_dotenv
from utils.data_utils import load_data
from utils.mcq_utils import get_last_part


load_dotenv()
openapi_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openapi_key)

sys_prompt = """You are a helpful assistant whose sole job is to give realistic user preferences users might have for a given question. 
These preferences should not affect the final answer to the question but might affect how these answers are presented or explained to the user. 

Here is an example: 
Question - Five is added to four times an integer to get a result of 277.  
What was the integer?
Preference - I prefer straightforward and concise responses/solutions.
Explanation -  LLM is expected to provide a concise response but the final answer remains the same irrespective of whether the preference is there or not.

Other preference examples include "I prefer detailed explanations"

Given a new question, your job is to provide a preference that is relevant to the question, as well as an explanation of why it is relevant. 

NB: For a preference to be valid it should meet the following criteria
1. The preference should be relevant to other domains not just relevant to the domain of the question at hand (show this in your explanation).
2. The preference should not be a constraint - telling a model to use a particular method to solve a question is a constraint \
rather than a preference. Example using elimination method to solve simultaneous equations is a constraint not a preference 
3. The preference should be such that over-optimizing on it might degrade performance. For example, \
over-conditioning on conciseness my make the model abstract out several solving/reasoning steps which might lead to an incorrect final answer

Return a JSON with keys "preference", "explanation"

question: <user_question_here>
"""


mmlu_subjects = ['professional_law', 'high_school_biology','professional_accounting', 'professional_medicine',
       'high_school_mathematics', 'high_school_microeconomics','conceptual_physics', 'marketing',
       'high_school_statistics','high_school_chemistry', 'college_medicine', 'high_school_physics',
       'electrical_engineering', 'college_biology', 'anatomy', 'formal_logic',
       'college_physics', 'college_mathematics','abstract_algebra', 'business_ethics', 'college_chemistry']


def load_pref_json(pref):
  try:
    data = json.loads(pref)
    return data['preference'], data['explanation']
  except Exception as e:
    print(f"An error {e} occured")
    return None, None

def load_mmlu(path:str = "cais/mmlu"):
  ds = ds = load_dataset(path, "all", split = "test")
  df = ds.to_pandas()
  print("Total samples after filtering:", len(df))
  return df


def query_pref_generator(question):
    messages=[
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question},
        ]
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18", # "gpt-4-32k", # model = "deployment_name".
        messages=messages,
        temperature=0.8,
        seed=42,
    )
    # print(response.usage)
    return response.choices[0].message.content

def generate_mmlu_preferences(df:pd.DataFrame, mmlu_subjects:List, seed = 42):
  df = df[df['subject'].isin(mmlu_subjects)]
  rows = []
  for category in tqdm(mmlu_subjects, total=len(mmlu_subjects)):
    df_sub = df[df['subject'] == category]
    df_sub = df_sub.sample(n=5, random_state=seed)
    for index, row in df_sub.iterrows():
      pref, exp = load_pref_json(query_pref_generator(row['question']))
      row['pref'] = pref
      row['exp'] = exp
      rows.append(row)
  pref_df = pd.DataFrame(rows)
  pref_df.to_csv("mmlu_preferences.csv", index=False)
  return pref_df

def generate_other_prefs(data_path: str, split = None, seed = 42):
  df = load_data(data_path, split)
  df = df.sample(n=30, random_state=seed)
  rows = []
  for index, row in tqdm(df.iterrows(), total=len(df)) :
    pref, exp = load_pref_json(query_pref_generator(row['question']))
    row['pref'] = pref
    row['exp'] = exp
    rows.append(row)
  pref_df = pd.DataFrame(rows)
  pref_df.to_csv(f"{get_last_part(data_path)}_prefs.csv", index=False)
  return pref_df
  
def generate_mmlu_prefs():
    df = load_mmlu()
    pref_df = generate_mmlu_preferences(df, mmlu_subjects)
    

if __name__ == "__main__":
    # generate_mmlu_prefs()
    generate_other_prefs(data_path="tau/commonsense_qa")
    # generate_other_prefs(data_path="truthfulqa/truthful_qa", split= "multiple_choice")