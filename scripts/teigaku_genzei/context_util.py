
import pandas as pd

def compose_text_block(sr: pd.Series) -> str:
    block = sr["Title"] + "\n" + sr["Answer"].removeprefix("[Ａ]")
    return block

def compose_context(section: int, df: pd.DataFrame) -> str:
    # print(df.index.name)
    block_list = [compose_text_block(row_sr) for (idx, row_sr) in df.iterrows()]
    context = "\n\n".join(block_list)
    return context

def compose_glossary(df: pd.DataFrame) -> str:
    term_desc_list = [
        (sr["Term"] + " " + sr["Description"]) for (idx, sr) in df.iterrows()
    ]
    ret = "\n".join(term_desc_list)
    return ret
