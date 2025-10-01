#!/bin/sh

if [ -z "$1" ]; then
    echo "./test_all.sh {model_name}"
    echo "e.g., ./test_all.sh granite-3.3-8b-instruct-teigaku-genzei-interp"
    exit 1
fi
MODEL_TEST_NAME="$1"
# MODEL_TEST_NAME=granite-3.3-8b-instruct-teigaku-genzei-interp
# MODEL_TEST_NAME=granite-3.3-8b-instruct-teigaku-genzei-ibm-v2-interp

# JUDGE_MODULE="metrics.llm_as_judge.direct.rits.llama3_3_70b"
# JUDGE_MODULE="metrics.llm_as_judge.direct.rits.llama4_maverick"
# JUDGE_MODULE="metrics.llm_as_judge.direct.rits.phi_4"
# JUDGE_MODULE="metrics.llm_as_judge.direct_positional_bias.rits.llama3_3_70b"
JUDGE_MODULE="metrics.llm_as_judge.direct_positional_bias.rits.phi_4"

TEST_DIR=./qa_table
TEST_NAMES="0024001-021 0024004-072_01"
MODEL_TEST_PATH=/proj/instructlab/checkpoints/${MODEL_TEST_NAME}
OUT_DIR=./tmp/${JUDGE_MODULE}__${MODEL_TEST_NAME}

mkdir ${OUT_DIR}

for TEST in ${TEST_NAMES}; do
    TEST_FILE=${TEST_DIR}/${TEST}.csv
    OUT_RESULT_PREFIX=${OUT_DIR}/result_${TEST}
    OUT_GLOBAL_PREFIX=${OUT_DIR}/global_${TEST}

    python ./test_qa.py \
        --test_file ${TEST_FILE} \
        --out_result_file ${OUT_RESULT_PREFIX}_after.csv \
        --out_global_score_file ${OUT_GLOBAL_PREFIX}_after.csv \
        --judge_module ${JUDGE_MODULE} \
        --model_path ${MODEL_TEST_PATH}
    python ./test_qa.py \
        --test_file ${TEST_FILE} \
        --out_result_file ${OUT_RESULT_PREFIX}_before.csv \
        --out_global_score_file ${OUT_GLOBAL_PREFIX}_before.csv \
        --judge_module ${JUDGE_MODULE} \
        --model_path ibm-granite/granite-3.3-8b-instruct
done
