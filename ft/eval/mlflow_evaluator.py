import mlflow

from ft.consts import EVAL_INPUT_COLUMN, EVAL_OUTPUT_COLUM


class ModelEvaluator():

    def __init__(self) -> None:
        pass

    @staticmethod
    def evaluate_model(model_info, eval_df, experiment_id, run_id, eval_target_column_name=EVAL_OUTPUT_COLUM):
        
        with mlflow.start_run(experiment_id=experiment_id, run_id=run_id):
            results = mlflow.evaluate(
                model_info.model_uri,
                eval_df,
                evaluators="default",
                model_type="text",    # parametrize this to support other types such as QnA, summarization etc.
                targets=eval_target_column_name,  # we do not set a target for a text-generation pipeline
                evaluator_config={"col_mapping": {"inputs": EVAL_INPUT_COLUMN}},
                extra_metrics=[mlflow.metrics.latency(), mlflow.metrics.exact_match(), mlflow.metrics.rouge1(), mlflow.metrics.rougeLsum(),
                               mlflow.metrics.rougeL()]
            )
        return results
