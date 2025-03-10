import string

from utils.model_utils import load_mcq_extractor_model


model, tokenizer =  load_mcq_extractor_model()

def get_response_template():
    return " ### Answer:"

def extract_references(ref_list):
    ref_text = " \nReferences: " 
    # ref_list = ast.literal_eval(text)
    ref_len = len(ref_list)
    options = list(string.ascii_uppercase)[:ref_len]
    for i, option in enumerate(ref_list):
        ref_text = ref_text + f"\n{options[i]}. {option}."

    return ref_text

def create_extraction_instruction(response, references, label=None):
    ref_text = extract_references(references)
    prompt = ""
    prompt = prompt + response + ref_text + f"\n{get_response_template()}"
    if label:
        prompt = prompt + f"{label}"
        
    return prompt

def extract_answer_letter(response, references):
    """
    Extract answer letter Given a model response and references.
    Takes the mcq_extractor model and tokenizer
    """
    prompt = create_extraction_instruction(response, references)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate the answer
    outputs = model.generate(**inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)

    # Decode the generated output to get the predicted answer
    predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    # print(predicted_answer)
    predicted_label = predicted_answer[-1]

    try:
        if predicted_label not in list(string.ascii_uppercase):
            raise ValueError
    except ValueError:
        print(f"An invalid answer {predicted_answer} is generated")
        predicted_label = -1
    return predicted_label