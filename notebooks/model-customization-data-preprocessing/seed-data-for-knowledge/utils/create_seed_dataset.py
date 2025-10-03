# Standard
from pathlib import Path
import json
import re
from typing import List, Dict

# Third Party
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer
import yaml

def get_seed_dataset(chunks_path: Path, seed_examples_path: Path) -> Dataset:
    """
    Creates a seed dataset from a path
    Args:
        path (str):   Path to directory of qna.yaml and chunks
    Returns:
        ds (Dataset): Transformers Dataset to be used to create a jsonl
                      of seed data for the knowledge generation pipeline in
                      SDG.
    """
    if not chunks_path.is_dir():
        raise ValueError(f"Path to chunks {chunks_path} must be a directory")
    if not seed_examples_path.is_dir():
        raise ValueError(f"Path to seed examples {seed_examples_path} must be a directory")

    files = list(seed_examples_path.iterdir())
    has_qna = any(f.name == 'qna.yaml' for f in files)
    files = list(chunks_path.iterdir())
    has_chunks_jsonl = any(f.name == 'chunks.jsonl' for f in files)

    if not has_qna:
        raise ValueError(f"Seed examples dir {seed_examples_path} does not contain a qna.yaml")

    if not has_chunks_jsonl:
        raise ValueError(f"Chunks dir {chunks_path} does not contain a chunks.jsonl")

    ds = create_dataset_from_dir(chunks_path, seed_examples_path)

    return ds

def read_chunks(chunks_path: Path) -> Dict[str, str]:
    """
    Returns a dictionary with all of the chunks in a chunks.jsonl
    The chunks may originate from one or more different files
    Args:
        path (Path): Path to directory of chunks in a file called chunks.jsonl
    Returns:
        chunks_dict (Dict[str,str]: Dictionary with key of the original file name
                                    and a list of chunks as the value
    """
    chunks_jsonl_path = chunks_path / "chunks.jsonl"
    chunks_dict = {}

    with open(chunks_jsonl_path, 'r') as file:
        for line in file:
            entry = yaml.safe_load(line)
            orig_filename = entry.get("file")

            if orig_filename not in chunks_dict:
                chunks_dict[orig_filename] = []

            chunks_dict[orig_filename].append(entry.get("chunk"))

    return chunks_dict

def create_dataset_from_dir(chunks_path: Path, seed_examples_path: Path) -> Dataset:
    """
    Process a directory with chunks and a qna.yaml return a dataset.
    Args:
        path (Path): Path to directory of chunks and qna.yaml.
    Returns:
        Dataset: Dataset object.
    """

    qna_yaml_path = seed_examples_path / "qna.yaml"

    with open(qna_yaml_path, 'r') as f:
      qna_yaml = yaml.safe_load(f)

    # Check for required fields
    if not all(key in qna_yaml for key in ['document_outline', 'domain', 'seed_examples']):
        raise ValueError("qna.yaml file is missing document_outline, domain, or seed_examples fields")

    chunks_dict = read_chunks(chunks_path)
    
    datasets = []
    for filename in chunks_dict.keys():
      chunks = chunks_dict[filename]
      chunk_ds = Dataset.from_dict(
          {
              "document": chunks,
              "document_outline": [qna_yaml["document_outline"]]
              * len(chunks),
              "document_title": [filename] * len(chunks), # TODO: is this really a necessary field?
              "domain": [qna_yaml["domain"]] * len(chunks),
          }
      )
      chunk_ds_with_icls = add_icls(qna_yaml, chunk_ds)
      datasets.append(chunk_ds_with_icls)

    return safe_concatenate_datasets(datasets)

def safe_concatenate_datasets(datasets: list[Dataset]) -> Dataset:
    """
    Concatenate datasets safely, ignoring any datasets that are None or empty.
    Args:
        datasets (list[Dataset]): List of Dataset objects to concatenate.
    Returns:
        Dataset: Dataset object with concatenated datasets.
    """
    filtered_datasets = [ds for ds in datasets if ds is not None and ds.num_rows > 0]

    if not filtered_datasets:
        return None

    return concatenate_datasets(filtered_datasets)

def get_token_count(text, tokenizer):
    return len(tokenizer.tokenize(text))

def add_icls(qna_yaml: Dict[str, str], chunked_document: Dataset, max_token_count: int = 1024) -> Dataset:
    """
    Add the ICLS label to the dataset.
    Args:
        qna_yaml (Dict): object representing qna.yaml file.
        dataset (Dataset): Dataset object.
    Returns:
        Dataset: Dataset object with ICLS label.
    """
    # TODO: make the tokenizer configurable at some level
    tokenizer = AutoTokenizer.from_pretrained("instructlab/granite-7b-lab")
    icl = qna_yaml["seed_examples"]
    chunked_document_all_icl = []
    for icl_ in icl:
        chunked_document_all_icl.append(
            chunked_document.map(
                lambda x: {
                    "icl_document": icl_["context"],
                    "icl_query_1": icl_["questions_and_answers"][0]["question"],
                    "icl_response_1": icl_["questions_and_answers"][0]["answer"],
                    "icl_query_2": icl_["questions_and_answers"][1]["question"],
                    "icl_response_2": icl_["questions_and_answers"][1]["answer"],
                    "icl_query_3": icl_["questions_and_answers"][2]["question"],
                    "icl_response_3": icl_["questions_and_answers"][2]["answer"],
                }
            )
        )
    chunked_document_all_icl = safe_concatenate_datasets(chunked_document_all_icl)
    def truncate_chunk(chunk: str):
        words = chunk.split()
        if len(words) > 7:
            return " ".join(words[:3]) + " ... " + " ".join(words[-3:])
        return chunk

    for c in chunked_document_all_icl:
        if get_token_count(c["document"], tokenizer) > max_token_count:
            raise ValueError(f"Chunk \"{truncate_chunk(c['document'])}\" exceeds token count of {max_token_count}")
    

    df = chunked_document_all_icl.to_pandas()
    new_ds = Dataset.from_pandas(df)

    # Only keep document greater than 100 tokens
    new_ds = new_ds.filter(
        lambda c: get_token_count(c["document"], tokenizer) > 100
    )
    return new_ds
