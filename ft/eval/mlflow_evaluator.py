import mlflow


class ModelEvaluator():

    def __init__(self) -> None:
        pass

    @staticmethod
    def evaluate_model(model_info, eval_df, eval_target_column_name="response"):
        with mlflow.start_run():
            results = mlflow.evaluate(
                model_info.model_uri,
                eval_df,
                evaluators="default",
                model_type="text",    # parametrize this to support other types such as QnA, summarization etc.
                targets=eval_target_column_name,
                evaluator_config={"col_mapping": {"inputs": "model_input"}}
            )
        return results
    