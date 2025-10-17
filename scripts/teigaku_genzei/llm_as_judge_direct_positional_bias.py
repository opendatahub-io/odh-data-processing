from typing import List, Dict, Any
from unitxt.llm_as_judge import LLMJudgeDirect, CriteriaWithOptions
from unitxt import add_to_catalog
from unitxt.inference import CrossProviderInferenceEngine, RITSInferenceEngine

from llm_as_judge_direct_rits_llama4 import RITSInferenceEngineFixed
import llm_as_judge_direct_rits_llama4

class LLMJudgeDirectPositionalBias(LLMJudgeDirect):
    # def prepare(self):
    #     super().prepare()
    #     print(f"LLMJudgeDirectPositionalBias.prepare() criteria: {self.criteria}")
    #     print(f"LLMJudgeDirectPositionalBias.prepare() inference_engine: {self.inference_engine}")
    #     if self.criteria is not None:
    #         self.ci_scores = [
    #             self.criteria.name,
    #             self.criteria.name + "_positional_bias_score",
    #             self.criteria.name + "_positional_bias_average",
    #         ]
    #         print(f"LLMJudgeDirectPositionalBias.prepare() ci_scores: {self.ci_scores}")

    POSTFIX_SCORE = "_positional_bias_score"
    POSTFIX_AVERAGE = "_positional_bias_average"
    POSTFIX_SELECTED_OPTION = "_positional_bias_selected_option"

    def compute(
        self,
        references: List[List[str]],
        predictions: List[str],
        task_data: List[Dict[str, Any]],
    ) -> List[Dict]:

        results = super().compute(references, predictions, task_data)

        forward_score_name = self.criteria.name
        backward_score_name = self.criteria.name + LLMJudgeDirectPositionalBias.POSTFIX_SCORE
        average_score_name = self.criteria.name + LLMJudgeDirectPositionalBias.POSTFIX_AVERAGE
        backward_selected_option_name = self.criteria.name + LLMJudgeDirectPositionalBias.POSTFIX_SELECTED_OPTION

        if self.criteria is not None:
            self.ci_scores = [
                forward_score_name,
                backward_score_name,
                average_score_name,
            ]
            print(f"LLMJudgeDirectPositionalBias.prepare() ci_scores: {self.ci_scores}")
            self.reduction_map = {"mean": [self.main_score] + self.ci_scores}

        pb_scores = [
            self.criteria.option_map[result[backward_selected_option_name]] 
            if (self.criteria is not None) and (self.criteria.option_map is not None) else 1
            for result in results
        ]
        pb_results = [
            result | {
                backward_score_name: pb_score,
                average_score_name: (result[self.main_score] + pb_score) / 2,
            }
            for result, pb_score in zip(results, pb_scores)
        ]
        return pb_results
    


def register(catalog_path: str) -> None:

    laaj_direct_rits_phi4 = LLMJudgeDirectPositionalBias(
        inference_engine=RITSInferenceEngine(
            max_tokens=1024,
            seed=42,
            temperature=0,
            model_name="microsoft/phi-4",
        ),
        evaluator_name="MICROSOFT_PHI_4",
        generate_summaries=False,
    )
    add_to_catalog(laaj_direct_rits_phi4, name="metrics.llm_as_judge.direct_positional_bias.rits.phi_4", overwrite=True, catalog_path=catalog_path)

    laaj_direct_rits_llama4_maverick = LLMJudgeDirectPositionalBias(
        inference_engine=RITSInferenceEngineFixed(
            max_tokens=1024,
            seed=42,
            temperature=0,
            model_name=llm_as_judge_direct_rits_llama4.MODEL_NAME,
            base_url=RITSInferenceEngineFixed.get_base_url_from_model_name_v(llm_as_judge_direct_rits_llama4.ENDPOINT_NAME, "/v1"),
            # base_url=RITSInferenceEngine.get_base_url_from_model_name(llm_as_judge_direct_rits_llama4.ENDPOINT_NAME) + "/v1",
        ),
        evaluator_name="LLAMA3_4_MAVERICK",
        generate_summaries=False,
    )
    add_to_catalog(laaj_direct_rits_llama4_maverick, name="metrics.llm_as_judge.direct_positional_bias.rits.llama4_maverick", overwrite=True, catalog_path=catalog_path)

