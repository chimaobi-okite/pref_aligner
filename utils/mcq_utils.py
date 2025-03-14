import string

def format_mcq_user_prompt(question, choices):
    ref_len = len(choices)
    user_prompt = f"""Which of the options best answers the question\n
    Question : {question}\nOptions: \n"""
    options = list(string.ascii_uppercase)[:ref_len]
    for option, choice in zip(options, choices):
        user_prompt += f"{option}. {choice}\n"
    return user_prompt


def format_system_prompt(profile = None):
    sys_prompt = f"""You are an AI assistant that provides factually accurate, unbiased, and helpful responses.\n"""
    if profile: 
        sys_prompt += f"""Here is the user preference: {profile}.
        Tailor your answer to thier perference.\n"""
    # sys_prompt += "Always remain truthful and unbiased."
    return sys_prompt


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

def get_last_part(s: str) -> str:
    """Splits a string by '/' and returns the last part. 
    If '/' is not present, returns the original string.
    """
    return s.split("/")[-1]

def calculate_math_accuracy(merged_df, is_same_column):
    return merged_df[is_same_column].eq("yes").sum() / len(merged_df)