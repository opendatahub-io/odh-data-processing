
# TODO: load YAML file.
# TODO: for each "questions_and_answers", select "question" and "answer".
# TODO: save as a CSV file.

import sys
import yaml
# import csv
import pandas as pd

input_file = sys.argv[1]
output_file = sys.argv[2]

# load YAML file
with open(input_file, "r") as file:
    data = yaml.load(file, Loader=yaml.FullLoader)
    # separate data
    print(data)
    qa_list = [
        {
            "Title": f"{isample + 1}-{iqa}",
            "Question": qa["question"], 
            "Answer": qa["answer"]
        }
        for (isample,sample) in enumerate(data["seed_examples"]) for (iqa, qa) in enumerate(sample["questions_and_answers"])
    ]
    df = pd.DataFrame(qa_list)
    df.to_csv(output_file, index=False, encoding="utf8") 

