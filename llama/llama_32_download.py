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
from utils_llama import read_prompt_list, get_completion_list, save_completion_list


#model_id = "meta-llama/Llama-3.2-1B"
model_id = "meta-llama/Llama-3.2-1B-Instruct"

pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    max_new_tokens=256,
)



################# promise binary random

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_random_0_bi.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_random_0_bi.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_random_5_bi.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_random_5_bi.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_random_10_bi.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_random_10_bi.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_random_20_bi.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_random_20_bi.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_random_40_bi.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_random_40_bi.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_random_80_bi.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_random_80_bi.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_random_160_bi.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_random_160_bi.txt'
save_completion_list(path, completion_list)


################# promise binary embedding

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_embedding_0_bi.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_embedding_0_bi.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_embedding_5_bi.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_embedding_5_bi.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_embedding_10_bi.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_embedding_10_bi.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_embedding_20_bi.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_embedding_20_bi.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_embedding_40_bi.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_embedding_40_bi.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_embedding_80_bi.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_embedding_80_bi.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_embedding_160_bi.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_embedding_160_bi.txt'
save_completion_list(path, completion_list)

################# promise binary tfidf

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_tfidf_0_bi.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_tfidf_0_bi.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_tfidf_5_bi.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_tfidf_5_bi.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_tfidf_10_bi.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_tfidf_10_bi.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_tfidf_20_bi.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_tfidf_20_bi.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_tfidf_40_bi.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_tfidf_40_bi.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_tfidf_80_bi.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_tfidf_80_bi.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_tfidf_160_bi.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_tfidf_160_bi.txt'
save_completion_list(path, completion_list)













################# promise multiclass random

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_random_0.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_random_0.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_random_5.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_random_5.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_random_10.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_random_10.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_random_20.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_random_20.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_random_40.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_random_40.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_random_80.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_random_80.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_random_160.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_random_160.txt'
save_completion_list(path, completion_list)


################# promise multi-class embedding

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_embedding_0.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_embedding_0.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_embedding_5.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_embedding_5.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_embedding_10.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_embedding_10.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_embedding_20.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_embedding_20.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_embedding_40.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_embedding_40.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_embedding_80.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_embedding_80.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_embedding_160.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_embedding_160.txt'
save_completion_list(path, completion_list)

################# promise multi-class tfidf

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_tfidf_0.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_tfidf_0.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_tfidf_5.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_tfidf_5.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_tfidf_10.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_tfidf_10.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_tfidf_20.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_tfidf_20.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_tfidf_40.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_tfidf_40.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_tfidf_80.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_tfidf_80.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_promise_tfidf_160.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_promise_tfidf_160.txt'
save_completion_list(path, completion_list)














################# nfr multiclass random

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_nfr_random_0.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_nfr_random_0.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_nfr_random_5.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_nfr_random_5.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_nfr_random_10.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_nfr_random_10.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_nfr_random_20.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_nfr_random_20.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_nfr_random_40.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_nfr_random_40.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_nfr_random_80.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_nfr_random_80.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_nfr_random_160.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_nfr_random_160.txt'
save_completion_list(path, completion_list)


################# nfr multi-class embedding

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_nfr_embedding_0.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_nfr_embedding_0.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_nfr_embedding_5.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_nfr_embedding_5.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_nfr_embedding_10.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_nfr_embedding_10.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_nfr_embedding_20.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_nfr_embedding_20.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_nfr_embedding_40.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_nfr_embedding_40.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_nfr_embedding_80.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_nfr_embedding_80.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_nfr_embedding_160.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_nfr_embedding_160.txt'
save_completion_list(path, completion_list)

################# nfr multi-class tfidf

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_nfr_tfidf_0.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_nfr_tfidf_0.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_nfr_tfidf_5.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_nfr_tfidf_5.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_nfr_tfidf_10.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_nfr_tfidf_10.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_nfr_tfidf_20.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_nfr_tfidf_20.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_nfr_tfidf_40.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_nfr_tfidf_40.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_nfr_tfidf_80.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_nfr_tfidf_80.txt'
save_completion_list(path, completion_list)

path = '/home/z004r5cc/re_genai/processed_prompts/prompt_nfr_tfidf_160.txt'
prompt_list = read_prompt_list(path)
completion_list = get_completion_list(pipe, prompt_list)
path = '/home/z004r5cc/re_genai/completions/llama31_8b/completion_nfr_tfidf_160.txt'
save_completion_list(path, completion_list)