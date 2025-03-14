import json
import os
import re
from typing import List
import pandas as pd
from tqdm import tqdm
from openai import AzureOpenAI
from openai import OpenAI

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

def math_evalutor_message(question, response, gold_answer):
    system_prompt = f"""You are provided with an llm response to a maths question, the math question as well as the final gold answer..
    Your job is to extract the final ultimate answer from the response in latex into $\\boxed{{answer}}$ format and also \
    tell if response and gold_answer are the same. return only a json with keys 'extracted_answer' and 'is_same'. 
    Where 'is_same' can take values of 'yes' or 'no' only. 
    
    NB: Do not attempt to answer or nor solve the question.

    ### question
    {question}

    
    ### response
    {response}

    ### Gold Answer
    {gold_answer}
    """
    messages=[
        {"role": "system", "content": system_prompt},
        ]
    return messages

def math_answer_extractor(question, response, gold_answer):
    messages = math_evalutor_message(question, response, gold_answer)
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18", # "gpt-4-32k", # model = "deployment_name".
        messages=messages,
        temperature=0,
        seed=42,
    )
    return response.choices[0].message.content, response.usage

def format_math_user_prompt(question:str):
    question = f"Solve the question: {question}"
    return question

def extract_last_boxed_answer(response):
    pattern = r"\\boxed\{"
    stack = []
    start_positions = []

    for match in re.finditer(pattern, response):
        start_positions.append(match.start())

    # Process each match to handle nested braces
    for start in start_positions:
        depth = 0
        content = []
        for i in range(start + len(r"\boxed{"), len(response)):
            char = response[i]
            if char == '{':
                depth += 1
            elif char == '}':
                if depth == 0:
                    break
                depth -= 1
            content.append(char)
        if depth == 0:
            stack.append(''.join(content))

    return stack[-1] if stack else None


def extract_json_from_extractor(json_str):
    try:
        # Extract the JSON-like content using regex
        match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON object found in the string.")
        
        json_content = match.group(0)  # Extracted JSON string
        
        # Parse JSON
        data = json.loads(json_content)

        # Extract values safely
        extracted_answer = data.get("extracted_answer", None)
        is_same = data.get("is_same", None)

        return extracted_answer, is_same

    except (json.JSONDecodeError, ValueError) as e:
        print(json_str)
        print(f"Error: {e}")
        return None, None

def calculate_math_accuracy(merged_df, is_same_column):
    return merged_df[is_same_column].eq("yes").sum() / len(merged_df)