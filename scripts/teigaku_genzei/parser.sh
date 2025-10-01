#!/bin/sh
source="source/"
output="docparser/"
# code_path="../sdg_hub/scripts/docparser_v2.py"
code_path="../sdg_hub/examples/knowledge_tuning/instructlab/docparser_v2.py"
python ${code_path}  --input-dir ${source} --output-dir ${output}
