
### functions that are used to support few-shot prompting
### @ yongjian.tang@tum.de

### save and read the generated completions from LLMs
### functions to process the completions, filter unnecessary
### functions to evaluate the processed completions 


from sklearn.metrics import precision_score, recall_score, f1_score



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



def read_completion_list(path):
    """read the saved list of prompts 
    input
        path, string path to save the list
    """
    with open(path, 'r') as file:  
        content = file.read()  
    completion_list_read = content.split('\n\n\n')   

    return completion_list_read



def get_evaluation_results(ground_truth, processed_completion_list):
    """input
        ground_truth: list, true requirement classifications
        processed_completion_list: list, completions from LLMs
    """
    # Calculate precision, recall, and F1 score  
    precision = precision_score(ground_truth, processed_completion_list, average='macro')  
    recall = recall_score(ground_truth, processed_completion_list, average='macro')  
    f1 = f1_score(ground_truth, processed_completion_list, average='macro')  
    
    # Print the results  
    print("Precision:", precision)  
    print("Recall:", recall)  
    print("F1 Score:", f1)  

    return precision, recall, f1



def process_completion_list(completion_list):
    """
    """
    processed_completion_list = [completion.split()[0] for completion in completion_list]

    return 


