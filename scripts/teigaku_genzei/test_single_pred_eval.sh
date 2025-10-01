#!/bin/sh

HELP_TEXT1="./test_single.sh {is_prediction_only} {model_path}\n" 
HELP_TEXT2="e.g., ./test_all.sh True path_to/granite-3.3-8b-instruct-teigaku-genzei-interp"
if [ -z "$1" ]; then
    echo ${HELP_TEXT1}
    echo ${HELP_TEXT2}
    exit 1
fi
if [ -z "$2" ]; then
    echo ${HELP_TEXT1}
    echo ${HELP_TEXT2}
    exit 1
fi
IS_PREDICTION_ONLY="$1"
MODEL_TEST_PATH="$2" # /proj/instructlab/checkpoints/${MODEL_TEST_NAME}
MODEL_TEST_NAME=`basename ${MODEL_TEST_PATH}`
# MODEL_TEST_NAME=granite-3.3-8b-instruct-teigaku-genzei-interp
# MODEL_TEST_NAME=granite-3.3-8b-instruct-teigaku-genzei-ibm-v2-interp

# JUDGE_MODULE="metrics.llm_as_judge.direct.rits.llama3_3_70b"
# JUDGE_MODULE="metrics.llm_as_judge.direct.rits.llama4_maverick"
# JUDGE_MODULE="metrics.llm_as_judge.direct.rits.phi_4"
# JUDGE_MODULE="metrics.llm_as_judge.direct_positional_bias.rits.llama3_3_70b"
# JUDGE_MODULE="metrics.llm_as_judge.direct_positional_bias.rits.llama4_maverick"
JUDGE_MODULE="metrics.llm_as_judge.direct_positional_bias.rits.phi_4"

DATA_CLASSIFICATION="public"
# DATA_CLASSIFICATION="proprietary"

TEST_DIR=./qa_table
TEST_NAMES="0024001-021"
OUT_DIR=./tmp/single_${JUDGE_MODULE}__${MODEL_TEST_NAME}
PREDICTION_FILE=${OUT_DIR}/prediction.jsonl

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
        --data_classification ${DATA_CLASSIFICATION} \
        --prediction_only ${IS_PREDICTION_ONLY} \
        --prediction_file ${PREDICTION_FILE} \
        --model_path ${MODEL_TEST_PATH}
    # python ./test_qa.py \
    #     --test_file ${TEST_FILE} \
    #     --out_result_file ${OUT_RESULT_PREFIX}_before.csv \
    #     --out_global_score_file ${OUT_GLOBAL_PREFIX}_before.csv \
    #     --judge_module ${JUDGE_MODULE} \
    #     --model_path ibm-granite/granite-3.3-8b-instruct
done
