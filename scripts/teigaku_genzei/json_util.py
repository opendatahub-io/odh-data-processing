
from typing import cast, Any, Hashable, Sequence, Collection
import re
import json

def is_title(text: str) -> bool:
    return re.match(r"[0-9０-９]+[－ー][0-9０-９]+", text) is not None

def get_title_match(text: str) -> re.Match | None:
    return re.match(r"([0-9０-９]+)[－ー]([0-9０-９]+)", text)

def is_qhead(text: str) -> bool:
    return re.match(r"^問", text) is not None

def is_ahead(text: str) -> bool:
    return re.match(r"^\[Ａ\]", text) is not None

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def write_jsonl_file(file_path: str, data_list: list[Any]):
    with open(file_path, 'w') as file:
        for data in data_list:
            file.write(json.dumps(data, ensure_ascii=False) + '\n')
            # json.dump(data, file, ensure_ascii=False)
    return

def read_jsonl_file(file_path: str) -> list[Any]:
    with open(file_path, 'r') as file:
        data_list = [
            json.loads(line) for line in file
        ]
        return data_list
