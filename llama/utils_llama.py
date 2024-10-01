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
    completion = pipeline(
        messages,
        max_new_tokens=6666,
    )        

    return completion[0]["generated_text"][-1]["content"]



def get_completion_list(pipeline, prompt_list):
    start_time = time.time()  
    completion_list = []
    for i, prompt in enumerate(prompt_list): 
        #print ("{}th prompt: ".format(i) + prompt + "\n######")
        completion = get_completion (pipeline, prompt)
        print(i, completion)
        #print("\n######")
        completion_list.append(completion)
    end_time = time.time()  
    print("execution time: {}".format(end_time - start_time) + "second")

    return completion_list


def save_completion_list(path, completion_list):
    """save the constructed prompts with few-shot examples in a list 
    input
        path, string path to save the list
        prompt_list, list of constructed prompts from the first requirement/sentence in test dataset to the last one
    """
    with open(path, 'w', newline='\n') as file:  
        for i, completion in enumerate(completion_list):
            if i+1 == len(completion_list):
                file.write(completion)
            else:     
                file.write(completion + "\n\n\n")

    print("### save" + path)            



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