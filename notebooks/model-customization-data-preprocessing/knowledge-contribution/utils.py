from openai import OpenAI
from pathlib import Path
import json
import re
from typing import Any, List
import yaml
import random
from textwrap import wrap

def get_random_chunks(chunks_jsonl_path: Path, num_chunks: int) -> List:
    """
    Creates a seed dataset from a path
    Args:
        chunks_jsonl_path (Path):       Path to the chunks.jsonl file
        num_chunk (int):                Number of chunks user wishes to randomly select
    Returns:
        selected_chunks_file_path (pathlib.Path): Path to the generated seed example file
    """
    if not chunks_jsonl_path.exists():
        raise ValueError(f"chunks.jsonl does not exist but should at {chunks_jsonl_path}")

    chunks = []

    with open(chunks_jsonl_path, 'r') as file:  # khaled was here
        for line in file:
            chunk = json.loads(line)
            chunks.append(chunk.get('chunk'))

    random_chunks = random.sample(chunks, num_chunks)

    return random_chunks

def create_knowledge_qna_yaml(output_dir: Path, contribution_domain: str, contribution_summary: str, seed_examples: List[dict]) -> Path:
    def str_presenter(dumper, data):
      if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
      elif len(data) > 80:
        data = "\n".join(wrap(data, 80))
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
      return dumper.represent_scalar('tag:yaml.org,2002:str', data)
    
    yaml.add_representer(str, str_presenter)
    
    # to use with safe_dump:
    yaml.representer.SafeRepresenter.add_representer(str, str_presenter)

    class IndentedDumper(yaml.Dumper):
        def increase_indent(self, flow=False, indentless=False):
            return super(IndentedDumper, self).increase_indent(flow, False)
    
    qna_output_path = output_dir / "qna.yaml"
    knowledge_contribution = {"domain": contribution_domain, "document_outline": contribution_summary, "seed_examples": seed_examples}
    
    with open(qna_output_path, 'w') as yaml_file:
        yaml.dump(knowledge_contribution, yaml_file, Dumper=IndentedDumper, default_flow_style=False, sort_keys=False, width=80)

    return qna_output_path


def parse_llm_json(response_text: str) -> Any:
    """
    Safely parse JSON returned by an LLM.
    
    Args:
        response_text (str): The text returned from an OpenAI chat completion.

    Returns:
        Any: Parsed Python object (list/dict). 
             If parsing fails, returns the raw string inside a dict: {"raw": response_text}.
    """
    # First try direct JSON parsing
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # If there's extra text, try extracting the JSON object/array
    match = re.search(r'(\{.*\}|\[.*\])', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Fallback: return raw text for debugging
    return {"raw": response_text}

def generate_qa_pairs(prompt: str, context: str, api_key: str, base_url: str = "https://api.openai.com/v1", model_name: str = "gpt-4o-mini", temperature: float = 0.0) -> dict:
    """
    Given a paragraph of text, call an OpenAI-compatible endpoint to
    generate 3 question-answer pairs.

    Args:
        paragraph (str): The input paragraph of text.
        api_key (str): Your API key for the endpoint.
        base_url (str): Base URL of the API (default: OpenAI).

    Returns:
        list: A list of dictionaries with "question" and "answer".
    """
    client = OpenAI(api_key=api_key, base_url=base_url)

    prompt = f"""
    {prompt}

    Text:
    {context}
    """

    response = client.chat.completions.create(
        model=model_name,
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
    )

    # Extract text from response
    content = response.choices[0].message.content.strip()
    qa_pairs = parse_llm_json(content)

    return qa_pairs

def review_seed_examples_file(seed_examples_path: Path, min_seed_examples: int = 5, num_qa_pairs: int = 3) -> None:
    """
    Review a seed example file has the expected number of fieldds
    Args:
        seed_examples_path (Path):      Path to the qna.yaml file
        min_seed_example (int):         Minimum number of expected seed examples
        num_qa_pairs (int):             Number of expected question and answer pairs in a seed example
    Returns:
        None
    """
    with open(seed_examples_path, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        errors = []
        print(f"Reviewing seed examples file at {seed_examples_path.resolve()}")

        # Check for document_outline
        if 'document_outline' not in yaml_data:
            errors.append("Missing contribution summary in 'document_outline'")
        else:
            # contribution summary is called document_outline internally
            print(f"Found contribution summary...")

        # Check for domain
        if 'domain' not in yaml_data:
            errors.append("Missing 'domain'")
        else:
            print(f"Found 'domain'...")

        # Check seed_examples
        seed_examples = yaml_data.get('seed_examples')
        if not seed_examples:
            errors.append("'seed_examples' section is missing or empty.")
        elif len(seed_examples) < min_seed_examples:
            errors.append(f"'seed_examples' should contain at least {min_seed_examples} examples, found {len(seed_examples)}. Please add {min_seed_examples - len(seed_examples)} more seed example(s)")
        else:
            print(f"Found {len(seed_examples)} 'contexts' in 'seed_examples'. Minimum expected number is {min_seed_examples}...")

        if seed_examples:
            for i, example in enumerate(seed_examples, start=1):
                qa_pairs = example.get('questions_and_answers')
                if not qa_pairs:
                    errors.append(f"Seed Example {i} is missing 'questions_and_answers' section.")
                elif len(qa_pairs) != num_qa_pairs:
                    errors.append(f"Seed Example {i} should contain {num_qa_pairs} question-answer pairs, found {len(qa_pairs)}. Please add {num_qa_pairs - len(qa_pairs)} more question-answer pair(s) to seed example {i}")
                else:
                    print(f"Seed Example {i} contains expected number ({num_qa_pairs}) of 'question_and_answers'...")

        if errors:
            print("\n\033[31mERROR! Seed Examples validation failed with the following issues:\033[0m")
            for err in errors:
                print(f"- {err}")
        else:
            print(f"\nSeed Examples YAML {seed_examples_path.resolve()} is valid :)")
        print(f"\n")