from pydantic import BaseModel
import pandas as pd
from typing import Dict, Any


class StartEvaluationRequest(BaseModel):
    adapter_path: str
    """
    Path to the adapter to be evaluated.
    """

    base_model_name: str
    """
    Name of the base model to be used for evaluation.
    """

    dataset_name: str

    """
    Name of the dataset to be used for evaluation.
    """


class EvaluationResponse(BaseModel):

    metrics: Dict[str, float]  # [str, Any] ?
    """
    Metrics calculated during evaluation.
    """

    csv: pd.DataFrame
    """
    A CSV representation of the results of the evaluation
    """

    class Config:
        arbitrary_types_allowed = True


if __name__ == "__main__":
    metrics = {'toxicity/v1/mean': 0.017866248597662584,
               'toxicity/v1/variance': 0.0011376551836664852,
               'toxicity/v1/p90': 0.04525991454720496,
               'toxicity/v1/ratio': 0.0,
               'flesch_kincaid_grade_level/v1/mean': 7.1866666666666665,
               'flesch_kincaid_grade_level/v1/variance': 12.741155555555554,
               'flesch_kincaid_grade_level/v1/p90': 11.32,
               'ari_grade_level/v1/mean': 11.246666666666666,
               'ari_grade_level/v1/variance': 46.558488888888895,
               'ari_grade_level/v1/p90': 19.9}
    csv = pd.DataFrame([{'text': 'This is a toxic comment', 'toxicity': metrics['toxicity/v1/mean']}])
    print(EvaluationResponse(metrics=metrics, csv=csv))
