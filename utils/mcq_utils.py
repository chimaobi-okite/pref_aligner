import string
from typing import List

def format_mcq_user_prompt(question, choices):
    ref_len = len(choices)
    user_prompt = f"""Which of the options best answers the question\n
    Question : {question}\nOptions: \n"""
    options = list(string.ascii_uppercase)[:ref_len]
    for option, choice in zip(options, choices):
        user_prompt += f"{option}. {choice}\n"
    return user_prompt


# def format_system_prompt(profile = None):
#     sys_prompt = f"""You are an AI assistant that provides factually accurate, unbiased, and helpful responses.\n"""
#     if profile: 
#         sys_prompt += f"""Here is the user preference: {profile}.
#         Tailor your answer to thier perference.\n"""
#     # sys_prompt += "Always remain truthful and unbiased."
#     return sys_prompt

def format_system_prompt(profile = None, examples = None, prompt_method=None):
    sys_prompt = f"""You are an AI assistant that provides factually accurate, unbiased, and helpful responses.\n"""
    if profile: 
        sys_prompt += f"""Here is the user preference: {profile}.
        Tailor your answer to their perference.\n"""
    if examples and prompt_method == 'icl':
        sys_prompt += "\n\nHere are some examples:\n"
        sys_prompt += examples.strip()  # Remove excess spacing if any
        
    if prompt_method == "cot":
        sys_prompt += """
        Here are some instructions:
        - Think step-by-step before answering.
        - Your response should be correct as well as aligned to the provided user preference when applicable
        """
    return sys_prompt


def format_higher_set_system_prompt(profile = None, examples = None, prompt_method=None):
    sys_prompt = f"""You are an AI assistant that provides factually accurate, unbiased, and helpful responses.\n"""
    if profile: 
        sys_prompt += f"""Here are the user preferences: {profile}.
        # Tailor your answer to relevant perferences.\n
        """
    if examples and prompt_method == 'icl':
        sys_prompt += "\n\nHere are some examples:\n"
        sys_prompt += examples.strip()  # Remove excess spacing if any
        
    if prompt_method == "cot":
        sys_prompt += """
        Here are some instructions:
        - Think step-by-step before answering.
        - Your response should be correct as well as align to only the relevant user preferences among all preferences
        when applicable
        """
    return sys_prompt

def build_pref_set(irrelevant_prefs: List[str], relevant_pref: str = None, position: int = None) -> List[str]:
    if position is None:
        position = len(irrelevant_prefs) // 2
    pref_list = irrelevant_prefs.copy()
    if relevant_pref:
        pref_list.insert(position, relevant_pref)
    return pref_list

def critic_message(question, response_to_q, preference):
    
    # crtic_sys_message= f"""Follow the instructions below to review your initial response to a  user query.
    
    # You are to return your output as a JSON object with the following structure:
    # {{"critic": "[Your critique here]","response": "[Your revised response here]"}}
    
    # Critique Request: Review your previous response to the user’s question in the last conversation turn.
    # Check if the response is factually accurate and adheres to or violates any user preferences stated earlier in the conversation that 
    # relate to this query. Provide a critique on how well those preferences were followed in 2 sentences.
    
    # User Preference: {preference}
    # Initial Question: {question}
    # Your intial response: {response_to_q}
    
    # Answer in this format:
    # Critic: [Your Critique Here]
    
    # Revision Request: Based on your critique, please rewrite your previous response to be factually accurate and aligned more closely
    # with the user’s stated preferences. Answer the question again:
    
    # Response: [Revised Response Here]
    # 
    # Return only the JSON dictionary.
    # """
    
    critic_sys_message = f"""Follow the instructions below to review your initial response to a user query.
    
    You are to return your output as a JSON object with the following structure:
    {{"critic": "[Your critique here]","response": "[Your revised response here]"}}

    Instructions:
    1. Review your previous response to the user's question in the last conversation turn.
    2. Check if the response is factually accurate and whether it aligns with the user's stated preferences.
    3. Provide a brief critique (2 sentences) on factual accuracy and preference alignment.
    4. Then, rewrite your response to better adhere to both factual correctness and the stated preference. Your revised response should also include the reference option.

    User Preference: {preference}
    Initial Question: {question}
    Your Initial Response: {response_to_q}

    Return only the JSON dictionary.
    """

    
    return critic_sys_message
    


def calculate_accuracy(merged_df, answer_column, gold_column='gold_option'):
    return (merged_df[answer_column] == merged_df[gold_column]).mean()


def get_answer_letter_option(answer, choices):
    """
    Given an answer string and list of options, extracts the letter option that answer string belongs to
    """
    ref_len = len(choices)
    answer_index = list(choices).index(answer)
    options = list(string.ascii_uppercase)[:ref_len]
    letter = options[answer_index]
    return letter


def get_answer_text(answer_key, choices):
    """
    Given an answer key and list of options, extracts the answer text
    """
    ref_len = len(choices)
    options = list(string.ascii_uppercase)[:ref_len]
    answer_index = options.index(answer_key)
    answer = choices[answer_index]
    return answer


def get_last_part(s: str) -> str:
    """Splits a string by '/' and returns the last part. 
    If '/' is not present, returns the original string.
    """
    return s.split("/")[-1]

def calculate_math_accuracy(merged_df, is_same_column):
    return merged_df[is_same_column].eq("yes").sum() / len(merged_df)

def aligner_agent_message(query, response, preference):
    aligner_message = f"""You are a preference aligner agent. Your task is to adjust a given response to better reflect a specified user preference, without re-answering the original query. 
    You are provided with the original query, the initial response from an answering agent, and a user preference.

    Only modify the response if the preference is relevant to the query or response. If the preference is irrelevant, return the original response unchanged.

    Query: {query}

    Initial Response: {response}

    User Preference: {preference}

    ###
    Return a JSON object with the following fields:
    - "response": the aligned response
    - "thoughts": a brief explanation of how (or whether) the response was aligned
    """
    return aligner_message
