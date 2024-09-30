### functions that are used to support few-shot prompting
### @ yongjian.tang@tum.de

import time

def read_prompt_list(path):
    """read the saved list of prompts 
    input
        path, string path to save the list
    """
    with open(path, 'r') as file:  
        content = file.read()  
    prompt_list_read = content.split('\n\n\n')   

    return prompt_list_read


def get_completion(pipeline, prompt):
    messages = [
        {"role": "system", "content": "You are a senior software engineer who is experienced in software requirement classification!"},
        {"role": "user", "content": prompt},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=6666,
    )        



def get_completion_list(pipeline, prompt_list):
    start_time = time.time()  
    completion_list = []
    for i, prompt in enumerate(prompt_list[:20]): 
        print ("{}th prompt: ".format(i) + prompt + "\n######")
        completion = get_completion (pipeline, prompt)
        print("completion: " + completion + "\n######")
        completion_list.append(completion)
    end_time = time.time()  
    print("execution time: {}".format(end_time - start_time) + "second")



# def get_completion_llama31(model, tokenizer, prompt):
    
#     messages = [
#         {"role": "system", "content": "You are a senior software engineer who is experienced in software requirement classification!"},
#         {"role": "user", "content": prompt},
#     ]

#     input_ids = tokenizer.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         return_tensors="pt"
#     ).to(model.device)

#     terminators = [
#         tokenizer.eos_token_id,
#         tokenizer.convert_tokens_to_ids("<|eot_id|>")
#     ]

#     outputs = model.generate(
#     input_ids,
#     max_new_tokens=6666,
#     eos_token_id=terminators,
#     do_sample=False,
#     temperature=0,
#     top_p=0.9,
#     )

#     response = outputs[0][input_ids.shape[-1]:]

#     return tokenizer.decode(response, skip_special_tokens=True)