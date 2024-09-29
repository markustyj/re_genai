### functions that are used to support few-shot prompting
### @ yongjian.tang@tum.de


# F: functional, A: availability, FT: fault tolerance, LF: look & feel, MN: maintainability, O: operational,
# PE: performance, PO: portability, SC: scalability, SE: security, US: usability, RE: reliability

# 0: availability, 1: performance, 2: maintainability, 3: portability, 4: scalability, 5: security, 6: fault tolerance

def prompt_construction(requirement_index, few_shot_list, num_shot, method, bi_classification, dataset):
    """

    """

    if bi_classification:
        prompt_base = ("Please classify the given software requirements into functional requirement or non-functional requirement. "
                "The answer should be in format {the given requirement: functional requirement or non-functional requirement}."
                )
    elif bi_classification == False:    
        prompt_base = ("Please classify the given software requirements into the following categories: "
                       "Functional, Availability, Fault Tolerance, Legal, Look and Feel, Maintainability, Operational, "
                       "Performance, Portability, Scalability, Security, Usability. "
                       "The answer should be in format {the given requirement: the name of classified categories}."
                        )
    else: 
        raise ValueError("bi_classification must be True or False.")    

    if dataset == "nfr":
        if bi_classification:
            raise ValueError("dataset name must be 'nfr' or 'promise'.")
        
        prompt_base = ("Please classify the given nonfunctional software requirements into the following categories: "
                       "Availability, Fault Tolerance, Maintainability, "
                       "Performance, Portability, Scalability, Security. "
                       "The answer should be in format {the given requirement: the name of classified categories}."
                        ) 


    if num_shot > 0:
        prompt = prompt_base + "The given requirement: "
    else:    
        if dataset == "nfr":
            few_shot_list = None
        else: # promise dataset
            few_shot_list = None   

        prompt == prompt_base + get_few_shot_examples(requirement_index, few_shot_list, num_shot, method, bi_classification) + "The given requirement: "


    return prompt       


def get_few_shot_examples(requirement_index, few_shot_list, num_shot, method, bi_classification):
    example_str = ""
    if method == "random":
        few_shot_list = 
        for i in range(num_shot):
            if bi_classification:       # the requirement                               # the category of this requirement
                example_str = example_str + few_shot_list[requirement_index][i][0] + ': ' + few_shot_list[requirement_index][i][2] + "\n"
            else: 
                example_str = example_str + few_shot_list[requirement_index][i][0] + ': ' + few_shot_list[requirement_index][i][1] + "\n"
    
    elif method == "embedding":
        for i in range(num_shot):
            if bi_classification:       # the requirement                               # the category of this requirement
                example_str = example_str + few_shot_list[requirement_index][i][0] + ': ' + few_shot_list[requirement_index][i][2] + "\n"
            else: 
                example_str = example_str + few_shot_list[requirement_index][i][0] + ': ' + few_shot_list[requirement_index][i][1] + "\n"

    elif method == "tfidf":
        for i in range(num_shot):
            if bi_classification:       # the requirement                               # the category of this requirement
                example_str = example_str + few_shot_list[requirement_index][i][0] + ': ' + few_shot_list[requirement_index][i][2] + "\n"
            else: 
                example_str = example_str + few_shot_list[requirement_index][i][0] + ': ' + few_shot_list[requirement_index][i][1] + "\n"

    return "Here are a few examples: \n" + example_str



def evaluate_the_dataset(loaded_data, num_shot, method):
    """ loaded_data is a N x 3 array for promise dataset, N x 2 array for nfr_so dataset
    """
    for sample in loaded_data:
        sample[0]