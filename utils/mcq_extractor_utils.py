import os
import ast
import string
from openai import AzureOpenAI
from openai import OpenAI
from together import Together

from utils.model_utils import load_mcq_extractor_model

openai_api_key = os.getenv("OPENAI_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")

#client = OpenAI(api_key=openai_api_key)
client = Together(api_key=together_api_key)

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
    invalid_count = 0

    try:
        if predicted_label not in list(string.ascii_uppercase):
            raise ValueError
    except ValueError:
        print(f"An invalid answer {predicted_answer} is generated")
        invalid_count = 1
        gpt_label = mcq_gpt_extractor(response, extract_references(references))
        print(gpt_label)
        final_label = extract_letter_from_json(gpt_label)
        print(final_label)
        predicted_label = final_label
    return predicted_label, invalid_count

def mcq_gpt_extractor(response, reference):
    system_prompt = """You are provided with a language model response to a multiple-choice question,
    along with the reference of answer choices for that question. Your job is to map the language model response
    to one of the choices. Just return the letter of the choice that matches the response in JSON format with key 'answer'."""
    
    example = """### Example
    response: If two modern organisms are distantly related in an evolutionary sense, then one should expect that they should share fewer homologous structures than two more closely related organisms. 
    
    References: A. they live in very different habitats. B. they should share fewer homologous structures than two more closely related organisms.
    C. their chromosomes should be very similar. D. they shared a common ancestor relatively recently.
    
    Answer: {{'answer': 'B'}}
    
    ### Your Task
    response: {response}
    {reference}
    
    Answer:"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example.format(response=response, reference=reference)}
    ]

    response = client.chat.completions.create(
        # model="gpt-4o-mini-2024-07-18", # "gpt-4-32k", # model = "deployment_name".
        #model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=messages,
        temperature=0,
        seed=42,
    )
    # print(response.usage)
    return response.choices[0].message.content


def extract_letter_from_json(text):
    option = -1
    try:
        answer_dict = ast.literal_eval(text)
        option = answer_dict.get("answer")
    except Exception as e:
        print(f"some error {e} occured in parsing")
    return option


def extract_answer_letters_batch(responses, list_of_references):
    """
    Extract answer letters for a batch of responses and reference lists.
    Runs everything in a single forward pass using batching.
    
    Args:
        responses (List[str]): List of model-generated responses
        list_of_references (List[List[str]]): List of options for each response
        
    Returns:
        List[str|int]: List of predicted labels like ['A', 'C', -1]
    """
    prompts = [
        create_extraction_instruction(response, references)
        for response, references in zip(responses, list_of_references)
    ]


    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=1,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    predicted_labels = []
    invalids = []
    for i, prediction in enumerate(decoded):
        label = prediction.strip()[-1]  # get last character
        if label in string.ascii_uppercase:
            predicted_labels.append(label)
            invalids.append(0)
        else:
            #print(f"Invalid answer generated: {prediction}")
            invalids.append(1)
            gpt_label = mcq_gpt_extractor(responses[i], extract_references(list_of_references[i]))
            #print(gpt_label)
            final_label = extract_letter_from_json(gpt_label)
            #print(final_label)
            predicted_labels.append(final_label)

    return predicted_labels, invalids