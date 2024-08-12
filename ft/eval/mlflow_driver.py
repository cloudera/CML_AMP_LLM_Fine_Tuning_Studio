from ft.eval.mlflow_evaluator import ModelEvaluator
from ft.eval.mlflow_logger import ModelLogger
from ft.eval.data_loader import Dataloader
from ft.pipeline import fetch_pipeline
import pandas as pd
from ft.eval.eval_job import EvaluationResponse
from ft.client import FineTuningStudioClient
from ft.api import *


def driver(
        dataset_id: str = None,
        base_model_id: str = None,
        adapter_id: str = None,
        client: FineTuningStudioClient = None):

    # TODO: remove hard-coded dependencies on GPU driver for evals
    device = "cuda"

    dataloader = Dataloader()
    logger = ModelLogger()
    evaluator = ModelEvaluator()

    # Load dataset
    eval_dataset, eval_column_name = dataloader.fetch_evaluation_dataset(dataset_id)

    # Get the model and adapter metadata.
    # given that this is a script that runs on a remote worker (not the same host
    # as the application), need to make gRPC calls to the app server.
    base_model: ModelMetadata = client.GetModel(GetModelRequest(id=base_model_id))
    adapter: AdapterMetadata = client.GetAdapter(GetAdapterRequest(id=adapter_id))

    # Load Model Pipeline
    # TODO: remove dependencies on model and adapter type
    pipeline = fetch_pipeline(base_model.huggingface_model_name, adapter.location, device=device)

    # Log model to MLFlow
    model_info = logger.log_model(pipeline)

    # Evaluate model
    results = evaluator.evaluate_model(model_info, eval_dataset, eval_column_name)

    results_df = pd.DataFrame(results.tables['eval_results_table'])
    response = EvaluationResponse(metrics=results.metrics, csv=results_df)
    return response
