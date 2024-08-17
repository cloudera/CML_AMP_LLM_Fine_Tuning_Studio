from ft.eval.mlflow_evaluator import ModelEvaluator
from ft.eval.mlflow_logger import ModelLogger
from ft.eval.data_loader import Dataloader
from ft.pipeline import fetch_pipeline
import pandas as pd
from ft.eval.eval_job import EvaluationResponse
from ft.client import FineTuningStudioClient
from ft.api import *
import json


def driver(
        dataset_id: str = None,
        base_model_id: str = None,
        adapter_id: str = None,
        bnb_config_id: str = None,
        generation_config_id: str = None,
        client: FineTuningStudioClient = None):

    # TODO: remove hard-coded dependencies on GPU driver for evals
    device = "cuda"

    dataloader = Dataloader()
    logger = ModelLogger()
    evaluator = ModelEvaluator()

    # Load dataset
    eval_dataset, eval_column_name = dataloader.fetch_evaluation_dataset(dataset_id, client=client)

    # Get the model and adapter metadata.
    # given that this is a script that runs on a remote worker (not the same host
    # as the application), need to make gRPC calls to the app server.
    base_model: ModelMetadata = client.GetModel(GetModelRequest(id=base_model_id))
    adapter: AdapterMetadata = client.GetAdapter(GetAdapterRequest(id=adapter_id))

    # Load in the generation config and bnb config.
    bnb_config_dict = json.loads(client.GetConfig(GetConfigRequest(
        id=bnb_config_id)).config.config) if bnb_config_id else None
    generation_config_dict = json.loads(client.GetConfig(GetConfigRequest(
        id=generation_config_id)).config.config) if generation_config_id else None

    # Load Model Pipeline
    # TODO: remove dependencies on model and adapter type. Right now this assumes that an adapter
    # is available in the project files location, and that the base model is available
    # on huggingface.
    assert base_model.type == MODEL_TYPE_HUGGINGFACE
    assert adapter.type == ADAPTER_TYPE_PROJECT
    pipeline = fetch_pipeline(
        base_model.huggingface_model_name,
        adapter.location,
        device=device,
        bnb_config_dict=bnb_config_dict,
        gen_config_dict=generation_config_dict
    )

    # Log model to MLFlow
    model_info = logger.log_model(pipeline)

    # Evaluate model
    results = evaluator.evaluate_model(model_info, eval_dataset, eval_column_name)

    results_df = pd.DataFrame(results.tables['eval_results_table'])
    response = EvaluationResponse(metrics=results.metrics, csv=results_df)
    return response
