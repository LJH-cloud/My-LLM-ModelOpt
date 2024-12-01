import json


def load_jsonl(file_path):
    list_data_dict = []
    with open(file_path, 'r') as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    prompts = []
    for sample in list_data_dict:
        # if sample['category'] == 'math':
        prompts += sample['turns']
    return prompts
