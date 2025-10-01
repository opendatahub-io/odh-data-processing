from unitxt.inference import CrossProviderInferenceEngine, RITSInferenceEngine
from unitxt.llm_as_judge import LLMJudgeDirect
from unitxt import add_to_catalog
from unitxt.llm_as_judge_constants import CriteriaOption, CriteriaWithOptions

# Source
# https://www.unitxt.ai/en/latest/catalog/catalog.metrics.llm_as_judge.direct.criteria.correctness_based_on_ground_truth.html
# This is a workaround to avoid the following issue.
# "correctness_base_on_ground_truth" post-process bug · Issue #7 · RYOKAWA/teigaku_genzei
# https://github.ibm.com/RYOKAWA/teigaku_genzei/issues/7

def register(catalog_path: str) -> None:

    correctness_based_on_ground_truth2 = CriteriaWithOptions(
        name="correctness_based_on_ground_truth2",
        description="Does the response correctly convey the same factual information as the ground truth?",
        options=[
            CriteriaOption(
                # name="correct",
                name="3",
                description="The response conveys the same factual meaning as the ground truth. Minor rewording, synonyms, or grammatical differences are acceptable. The response is relevant to the question and does not introduce unrelated or misleading information.",
            ),
            CriteriaOption(
                # name="incomplete",
                name="2",
                description="The response contains some correct information but is incomplete or lacks essential details. It may also contain minor inaccuracies or extraneous information that slightly misrepresents the ground truth.",
            ),
            CriteriaOption(
                # name="wrong",
                name="1",
                description="The response does not align with the ground truth. It either presents incorrect, unrelated, or misleading information, or omits key details that change the intended meaning.",
            ),
        ],
        option_map={
            # "correct": 1.0,
            # "incomplete": 0.5,
            # "wrong": 0.0,
            "3": 1.0,
            "2": 0.5,
            "1": 0.0,
        },
    )

    add_to_catalog(correctness_based_on_ground_truth2, name="metrics.llm_as_judge.direct.criteria.correctness_based_on_ground_truth2", overwrite=True, catalog_path=catalog_path)
