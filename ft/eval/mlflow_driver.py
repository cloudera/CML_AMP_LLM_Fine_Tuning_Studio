from ft.eval.mlflow_evaluator import ModelEvaluator
from ft.eval.mlflow_logger import ModelLogger
from ft.eval.data_loader import Dataloader
from ft.pipeline import fetch_pipeline
import pandas as pd
from ft.eval.eval_job import StartEvaluationRequest, EvaluationResponse


def driver(StartEvaluationRequest):

    dataset_name = StartEvaluationRequest.dataset_name
    base_model_name = StartEvaluationRequest.base_model_name
    adapter_name = StartEvaluationRequest.adapter_path
    device = "cuda"

    dataloader = Dataloader()
    logger = ModelLogger()
    evaluator = ModelEvaluator()

    # Load dataset
    eval_dataset, eval_column_name = dataloader.fetch_evaluation_dataset(dataset_name)
    # Load Model Pipeline
    pipeline = fetch_pipeline(base_model_name, adapter_name, device=device)

    # Log model to MLFlow
    model_info = logger.log_model(pipeline)

    # Evaluate model
    results = evaluator.evaluate_model(model_info, eval_dataset, eval_column_name)

    results_df = pd.DataFrame(results.tables['eval_results_table'])
    response = EvaluationResponse(metrics=results.metrics, csv=results_df)
    return response
