import json

def get_dataset_iterator(file_path):
    with open(file_path) as f:
        for i, item in enumerate(f):
            data = json.loads(item)
            yield {"id": i, "data": "question : " + data["question"] + " answer : " +  data["answer"]}

def get_prompt(entry_data, template):
    return template.format(problem=entry_data)