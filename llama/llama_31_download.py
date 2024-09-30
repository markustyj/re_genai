### functions that are used to support few-shot prompting
### @ yongjian.tang@tum.de

# F: functional, A: availability, FT: fault tolerance, LF: look & feel, MN: maintainability, O: operational,
# PE: performance, PO: portability, SC: scalability, SE: security, US: usability, RE: reliability

# 0: availability, 1: performance, 2: maintainability, 3: portability, 4: scalability, 5: security, 6: fault tolerance

import time
import torch
from transformers import pipeline
from utils_llama import read_prompt_list, get_completion_list, save_completion_list

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    max_new_tokens=6666,
)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_random_0_bi.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(prompt_list[:20], pipe)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_random_0_bi.txt'
save_completion_list(path, completion_list)


# path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_random_5_bi.txt'
# prompt_list = read_prompt_list(path)
# get_completion_list(prompt_list, pipe)


# path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_random_0.txt'
# prompt_list = read_prompt_list(path)
# get_completion_list(prompt_list, pipe)


# path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_random_5.txt'
# prompt_list = read_prompt_list(path)
# get_completion_list(prompt_list, pipe)
