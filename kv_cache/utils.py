import json
from collections import OrderedDict

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

class Output(OrderedDict):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        else:
            return self.to_tuple()[key]
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    def to_tuple(self):
        return tuple(self[k] for k in self.keys())