import ast
from typing import List
import torch

from enums import PromptMethod
from utils.janus_utils import apply_template_mistral_instruct, extract_after_inst
from utils.mcq_extractor_utils import extract_answer_letters_batch, extract_critic_and_response
from utils.mcq_utils import critic_message
from utils.utils import get_gemma_messages, get_messages

critic_sys_message = "You are an AI assistant that provides factually accurate, unbiased, and helpful responses."

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
    


def run_client_mcq_generation(user_prompts, batch_options_list, sys_messages, client, model,
                              preferences, prompt_method, is_user_message_model):
    
    if is_user_message_model:
        batched_inputs = [get_gemma_messages(sys_message=sys_message, user_message=user_prompt) 
                        for (sys_message, user_prompt) in zip(sys_messages, user_prompts)]
    else:
        batched_inputs = [get_messages(sys_message=sys_message, user_message=user_prompt) 
                    for (sys_message, user_prompt) in zip(sys_messages, user_prompts)]
        
    # print(batched_inputs)
    responses = []
    input_len = len(batched_inputs)
    for messages in batched_inputs:
        content = client_generate(client, model, messages=messages)
        responses.append(content)
    list_of_references = batch_options_list[:input_len]
    
    print(list_of_references)
    list_of_references = [ast.literal_eval(refs) for refs in list_of_references]
    print(list_of_references)
    
    critics = [None for refs in list_of_references]
    
    if prompt_method == PromptMethod.SELF_CRITIC.value:
        critic_responses = []
        critic_critics = []
        critic_messages = [critic_message(query, response_to_q, preference)
                           for query, response_to_q, preference
                           in zip(user_prompts, responses, preferences)]
        critic_messages = [get_messages(
            sys_message=critic_sys_message,user_message=message) for message in critic_messages]
        results = []
        for messages in critic_messages:
            content = client_generate(client, model, messages=messages)
            results.append(content)
        result_jsons = [extract_critic_and_response(response) for response in results]
        for i, response in enumerate(result_jsons):
            if response:
                critic_responses.append(response['response'])
                critic_critics.append(response['critic'] )
            else:
                print(results[i])
                critic_responses.append(responses[i])
                critic_critics.append("No critic provided")
            
        responses = critic_responses
        critics = critic_critics
        
    thoughts = ["t" for refs in list_of_references]
    predictions, invalids = extract_answer_letters_batch(responses, list_of_references)
    return responses, predictions, invalids, critics, thoughts

def run_mcq_generation(user_prompts, batch_options_list, sys_messages, generator, preferences,
                       prompt_method, is_user_message_model):
    if is_user_message_model:
        # batched_inputs = [f"{sys_message}\n\n{user_prompt}"
        #                   for sys_message, user_prompt in zip(sys_messages, user_prompts)]
        batched_inputs = [get_gemma_messages(sys_message=sys_message, user_message=user_prompt) 
                        for (sys_message, user_prompt) in zip(sys_messages, user_prompts)]
    else:
        batched_inputs = [get_messages(sys_message=sys_message, user_message=user_prompt) 
                        for (sys_message, user_prompt) in zip(sys_messages, user_prompts)]
    results = generator(batched_inputs, max_new_tokens=500,)
    input_len = len(batched_inputs)
    responses = [results[i][0]["generated_text"][-1]['content'] for i in range(0, input_len)]
    list_of_references = batch_options_list[:input_len]
    list_of_references = [ast.literal_eval(refs) for refs in list_of_references]
    critics = [None for refs in list_of_references]
    
    if prompt_method == PromptMethod.SELF_CRITIC.value:
        critic_responses = []
        critic_critics = []
        critic_messages = [critic_message(query, response_to_q, preference)
                           for query, response_to_q, preference
                           in zip(user_prompts, responses, preferences)]
        critic_messages = [get_messages(
            sys_message=critic_sys_message,user_message=message) for message in critic_messages]
        results = generator(critic_messages, max_new_tokens=1024,)
        results = [results[i][0]["generated_text"][2]['content'] for i in range(0, input_len)]
        result_jsons = [extract_critic_and_response(response) for response in results]
        for i, response in enumerate(result_jsons):
            if response:
                critic_responses.append(response['response'])
                critic_critics.append(response['critic'] )
            else:
                print(results[i])
                critic_responses.append(responses[i])
                critic_critics.append("No critic provided")
            
        responses = critic_responses
        critics = critic_critics
        
    thoughts = ["t" for refs in list_of_references]
    predictions, invalids = extract_answer_letters_batch(responses, list_of_references,)
    return responses, predictions, invalids, critics, thoughts

def run_janus_mcq_generation(user_prompts, batch_options_list, sys_messages, janus, janus_tokenizer,
                             preferences, prompt_method):
    input_strs = [apply_template_mistral_instruct(sys_message,user_prompt) for 
                  (sys_message, user_prompt) in zip(sys_messages, user_prompts)]
    input_len = len(input_strs)
    inputs = janus_tokenizer(input_strs, return_tensors="pt", padding=True, truncation=True).to(janus.device)
    output_ids = janus.generate(inputs['input_ids'], max_new_tokens=1024, temperature=0.0,do_sample=False,
                                pad_token_id=janus_tokenizer.eos_token_id)
    responses = janus_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    responses = [extract_after_inst(res) for res in responses]
    list_of_references = batch_options_list[:input_len]
    list_of_references = [ast.literal_eval(refs) for refs in list_of_references]
    
    critics = [None for refs in list_of_references]
    
    if prompt_method == PromptMethod.SELF_CRITIC.value:
        critic_responses = []
        critic_critics = []
        critic_messages = [critic_message(query, response_to_q, preference)
                           for query, response_to_q, preference
                           in zip(user_prompts, responses, preferences)]
        input_strs = [apply_template_mistral_instruct(critic_sys_message,message) 
                           for message in critic_messages]
        # print(input_strs)
        input_len = len(input_strs)
        inputs = janus_tokenizer(input_strs, return_tensors="pt", padding=True, truncation=True).to(janus.device)
        output_ids = janus.generate(inputs['input_ids'], max_new_tokens=2048, temperature=0.0,do_sample=False,
                                    pad_token_id=janus_tokenizer.eos_token_id)
        responses = janus_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        # print(responses)
        results = [extract_after_inst(res) for res in responses]
        # print(results)
        result_jsons = [extract_critic_and_response(response) for response in results]
        for i, response in enumerate(result_jsons):
            if response:
                critic_responses.append(response['response'])
                critic_critics.append(response['critic'] )
            else:
                print(results[i])
                critic_responses.append(responses[i])
                critic_critics.append("No critic provided")
            
        responses = critic_responses
        critics = critic_critics
        
        print(responses)
        # responses = [response['response'] for response in results]
        # critics = [response['critic'] for response in results]
        
        # print(responses)

        # print(responses)
        # print(critics)
        
    thoughts = ["t" for refs in list_of_references]
    predictions, invalids = extract_answer_letters_batch(responses, list_of_references)
    return responses, predictions, invalids, critics, thoughts

def run_qwen_generation(user_prompts, batch_options_list, sys_messages, model, tokenizer,
                        preferences, prompt_method, enable_thinking):
    batched_inputs = [get_gemma_messages(sys_message=sys_message, user_message=user_prompt) 
                        for (sys_message, user_prompt) in zip(sys_messages, user_prompts)]
    
    input_len = len(batched_inputs)
    
    list_of_references = batch_options_list[:input_len]
    list_of_references = [ast.literal_eval(refs) for refs in list_of_references]
    critics = ["a" for refs in list_of_references]
    # thoughts = ["t" for refs in list_of_references]

    responses, thoughts = qwen_batch_generate(batched_inputs, model, tokenizer, enable_thinking)
    predictions, invalids = extract_answer_letters_batch(responses, list_of_references)
    
    # print(thoughts)

    return responses, predictions, invalids, critics, thoughts

def qwen_batch_generate(batched_inputs, model, tokenizer, enable_thinking = True):
    input_strs = tokenizer.apply_chat_template(
        batched_inputs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )

    model_inputs = tokenizer(input_strs, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            do_sample=False  # or True if you want sampling
        )
    
    responses = []
    thoughts = []
    for i in range(len(input_strs)):
        input_len = len(model_inputs["input_ids"][i])
        output_ids = generated_ids[i][input_len:].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        responses.append(content)
        thoughts.append(thinking_content)

    return responses, thoughts