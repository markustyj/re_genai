import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import pickle
# import tiktoken
# import sys
# import transformers
# from transformers import pipeline, LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
# from gpt_ner_api_codes.prompts_all import *
# from timeit import default_timer
# import nltk
# nltk.download('punkt')

import torch
from transformers import pipeline

#model_id = "meta-llama/Llama-3.2-1B"
model_id = "meta-llama/Llama-3.2-1B-Instruct"

pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    max_new_tokens=256,
)

messages = [
    {"role": "system", "content": "You are a senior software engineer who is experienced in software requirement classification!"},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)

print(outputs[0]["generated_text"][-1])

