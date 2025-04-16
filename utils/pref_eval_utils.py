import argparse
import os
import re
import ast
import string
import pandas as pd
from openai import AzureOpenAI
from openai import OpenAI
from together import Together

openai_api_key = os.getenv("OPENAI_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")

client = OpenAI(api_key=openai_api_key)

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
        model="gpt-4o-mini-2024-07-18", # "gpt-4-32k", # model = "deployment_name".
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
                                main_path:str = "/content/robuset_main.csv") -> pd.DataFrame:
    df = pd.read_csv(df_path)
    robo_df = pd.read_csv(main_path)
    df['dup_id'] = df.groupby('question').cumcount()
    robo_df['dup_id'] = robo_df.groupby('question').cumcount()
    merged = df.merge(robo_df[['question', 'dup_id', 'options']], on=['question', 'dup_id'], how='inner')
    merged['options_x'] = None
    merged['options_x'] = merged['options_y']
    merged = merged.drop(columns=['options_y', 'dup_id'])
    merged = merged.rename(columns={'options_x': 'options'})
    return merged
