### functions that are used to support few-shot prompting
### @ yongjian.tang@tum.de

# F: functional, A: availability, FT: fault tolerance, LF: look & feel, MN: maintainability, O: operational,
# PE: performance, PO: portability, SC: scalability, SE: security, US: usability, RE: reliability

# 0: availability, 1: performance, 2: maintainability, 3: portability, 4: scalability, 5: security, 6: fault tolerance

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(model_id)


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


# messages = [
#     {"role": "system", "content": "You are a senior software engineer who is experienced in software requirement classification!"},
#     {"role": "user", "content": "Who are you?"},
# ]

# input_ids = tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     return_tensors="pt"
# ).to(model.device)

# terminators = [
#     tokenizer.eos_token_id,
#     tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

# outputs = model.generate(
#     input_ids,
#     max_new_tokens=6666,
#     eos_token_id=terminators,
#     do_sample=False,
#     temperature=0,
#     top_p=0.9,
# )

# response = outputs[0][input_ids.shape[-1]:]

# print(tokenizer.decode(response, skip_special_tokens=True))


def get_completion(model, tokenizer, prompt):
    
    messages = [
        {"role": "system", "content": "You are a senior software engineer who is experienced in software requirement classification!"},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
    input_ids,
    max_new_tokens=6666,
    eos_token_id=terminators,
    do_sample=False,
    temperature=0,
    #top_p=0.9,
    )

    response = outputs[0][input_ids.shape[-1]:]

    return tokenizer.decode(response, skip_special_tokens=True)




## load the prompt_list
def read_prompt_list(path):
    """read the saved list of prompts"""
    with open(path, 'r') as file:  
        prompt_list_read = []
        prompt = ""        
        for line in file:
            if line == "\n":
                prompt_list_read.append(prompt)
                prompt = ""
            else:
                prompt = prompt + "\n" + line  

    return prompt_list_read



def get_completion_list(prompt_list, tokenizer):
    start_time = time.time()  
    completion_list = []
    for i, prompt in enumerate(prompt_list[:20]): 
        print ("{}th prompt: ".format(i) + prompt + "\n######")
        completion = get_completion (model, tokenizer, prompt)
        print("completion: " + completion + "\n######")
        completion_list.append(completion)
    end_time = time.time()  
    print("execution time: {}".format(end_time - start_time) + "second")



path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_random_0_bi.txt'
prompt_list = read_prompt_list(path)
get_completion_list(prompt_list, tokenizer)


path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_random_5_bi.txt'
prompt_list = read_prompt_list(path)
get_completion_list(prompt_list, tokenizer)


path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_random_0.txt'
prompt_list = read_prompt_list(path)
get_completion_list(prompt_list, tokenizer)


path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_random_5.txt'
prompt_list = read_prompt_list(path)
get_completion_list(prompt_list, tokenizer)
