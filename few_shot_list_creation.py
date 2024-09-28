## create the few-shot list for each software requirement based on random selection, sentence embedding, and TF-IDF selection
## @ yongjian.tang@tum.de
import random
import csv

def load_csv(file_path):  
    """ load csv file line by line, in list format  -> [ [the first-row elements], [the second-row elements], [], ... ]"""
    data = []  
    with open(file_path, 'r') as file:  
        csv_reader = csv.reader(file)  
        for row in csv_reader:  
            data.append(row)  
    return data[1:]  # do not need the first row of title ['RequirementText', 'class']


def load_specific_dataset(dataset):
    """load the specific dataset using load_csv() method
    input
    dataset: the name of the processed datasets -> "nfr" or "promise"
    output
    loaded_data: the loaded csv file in format [ [the first-row elements], [the second-row elements], [], ... ]
    """
    if dataset == "nfr": 
        loaded_data = load_csv('/Users/yongjiantang/Desktop/tang/code/re_genai/data/processed_nfr_so.csv')
    elif dataset == "promise":
        loaded_data = load_csv('/Users/yongjiantang/Desktop/tang/code/re_genai/data/processed_promise.csv')

    return loaded_data



def get_random_few_shot_list(dataset):
    """
    input
    dataset: the name of the processed datasets -> "nfr" or "promise"
    output
    few_shot_list: three-dimensional list --> each requirement sentence, its k closest few-shot examples, [the few-shot examples, the multi-class, the binary-class] 
    """
    loaded_data = load_specific_dataset(dataset)
    num_k_closest_examples = 160

    few_shot_list = []
    k_closest_examples = []

    # create the list of k closest few-shot examples
    values = list(range(0, len(loaded_data)))
    random.shuffle(values)
    for i in range(num_k_closest_examples):
        index = values.pop()
        k_closest_examples.append(loaded_data[index])

    # repeat and reuse this list of k closest few-shot examples for each software requirement, i.e. same list for each requirement
    for i in range(len(loaded_data)):
        few_shot_list.append(k_closest_examples)
    
    return few_shot_list



def get_random_few_shot_list(dataset):
    """
    input
    dataset: the name of the processed datasets -> "nfr" or "promise"
    output
    few_shot_list: three-dimensional list --> each requirement sentence, its k closest few-shot examples, [the few-shot examples, the multi-class, the binary-class] 
    """
    loaded_data = load_specific_dataset(dataset)
    num_k_closest_examples = 160

    few_shot_list = []
    k_closest_examples = []

    