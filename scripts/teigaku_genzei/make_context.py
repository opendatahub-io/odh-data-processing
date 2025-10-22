import sys
from typing import cast, Any, Hashable, Sequence, Collection
import json
import argparse

import pandas as pd

import json_util

from context_util import compose_context, compose_glossary

# TODO: input: qa.csv
# TODO: input: glossary.csv
# TODO: output: context.csv

# TODO: extract section numbers
# TODO: collect entries of same section numbers.

ARG_OPTION="glossary_option"
ARG_INPUT_QA_FILE="qa_file"
ARG_INPUT_GLOSSARY_FILE="glossary_file"
ARG_OUTPUT_CONTEXT_FILE="out_context_file"
OPT_GLOSSARY_APPENDIX="appendix"
OPT_GLOSSARY_HEADER="header"
OPT_GLOSSARY_NONE="none"

def config():
    # print("Usage: python script.py <option> <input_qa.csv> <input_glossary.csv> <output_context.csv")
    parser = argparse.ArgumentParser(description="Make context data from QAs and glossaries.")
    parser.add_argument(
        '--' + ARG_OPTION, 
        type=str, 
        default=OPT_GLOSSARY_APPENDIX, 
        choices=[OPT_GLOSSARY_APPENDIX, OPT_GLOSSARY_HEADER, OPT_GLOSSARY_NONE],
    )
    parser.add_argument('--' + ARG_INPUT_QA_FILE, type=str, required=True, metavar="input_qa.csv")
    parser.add_argument('--' + ARG_INPUT_GLOSSARY_FILE, type=str, required=True, metavar="input_glossary.csv")
    parser.add_argument('--' + ARG_OUTPUT_CONTEXT_FILE, type=str, required=True, metavar="output_context.csv")

    args = parser.parse_args()

    args = vars(args)
    
    return args

def extract_section_number(title: str) -> tuple[int, int]:
    match = json_util.get_title_match(title)
    if match is not None:
        section_num_str = match.group(1)
        subsection_num_str = match.group(2)
        return (int(section_num_str), int(subsection_num_str))
    return (0, 0)

def main()-> None:

    args = config()

    option = args[ARG_OPTION]
    input_qa_path = args[ARG_INPUT_QA_FILE]
    input_glossary_path = args[ARG_INPUT_GLOSSARY_FILE]
    output_context_path = args[ARG_OUTPUT_CONTEXT_FILE]

    qa_df = pd.read_csv(input_qa_path, encoding="utf8")
    glossary_df = pd.read_csv(input_glossary_path, encoding="utf8")

    section_df = qa_df.apply(lambda x: pd.Series(extract_section_number(x["Title"]), index=["section", "subsection"]), axis=1)
    print(section_df)
    qas_df = pd.concat([qa_df, section_df], axis=1)
    section_gp = qas_df.groupby("section")
    # context_df = section_gp.agg(lambda df: compose_context(1, df))
    glossary_str = "用語集\n" + compose_glossary(glossary_df) + "\n\n"

    (header, appendix, a_section, a_qindex) = (
        ("", [glossary_str], [-1], [-1]) if option == OPT_GLOSSARY_APPENDIX else 
        (glossary_str, [], [], []) if option == OPT_GLOSSARY_HEADER else 
        ("", [], [], [])
    )
    context_list = [header + compose_context(cast(int, section), df) for (section, df) in section_gp] + appendix
    section_list = [cast(int, section) for (section, df) in section_gp] + a_section
    qindex_list = [cast(int, df.index[0]) for (section, df) in section_gp] + a_qindex
    qlist_list = [json.dumps(df.index.to_list()) for (section, df) in section_gp] + [json.dumps([qi]) for qi in a_qindex]
    # qlist_list = [json.dumps(df.index.to_list()) for (section, df) in section_gp] + a_qindex
    # for section, df in section_gp:
    #     print(section)
    #     print(df)
    out_df = pd.DataFrame({"section": section_list, "qindex": qindex_list ,"qlist": qlist_list, "context": context_list})
    # out_df = context_df
    out_df.to_csv(output_context_path, index=False, encoding="utf8")

    return

if __name__ == "__main__":

    main()
