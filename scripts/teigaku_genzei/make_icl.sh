#!/bin/sh

python ./make_icl.py --short_context True --qa_file qa_table/0024004-072_01.csv --glossary_file qa_table/0024004-072_01_glossary.csv --context_file context/0024004-072_01_context.csv --out_icl_file icl/icl.jsonl
