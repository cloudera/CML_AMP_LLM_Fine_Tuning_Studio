from ft.eval.mlflow_evaluator import ModelEvaluator
from ft.eval.mlflow_logger import ModelLogger
from ft.eval.data_loader import Dataloader
from ft.eval.mlflow_pyfunc import MLFlowTransformers
from ft.pipeline import fetch_pipeline
import pandas as pd
from ft.eval.eval_job import EvaluationResponse
from ft.client import FineTuningStudioClient
from ft.api import *
import json
import time
import torch
from typing import List
from ft.consts import EVAL_INPUT_COLUMN, EVAL_OUTPUT_COLUM


def table_fetcher(results):
    json_uri = results.artifacts['eval_results_table'].uri
    json_obj = json.load(open(json_uri))
    columns = json_obj["columns"]
    data = json_obj["data"]
    df = pd.DataFrame(data, columns=columns)
    return df


def driver(
        dataset_id: str = None,
        base_model_id: str = None,
        adapter_id: str = None,
        bnb_config_id: str = None,
        generation_config_id: str = None,
        prompt_id: str = None,
        selected_features: List[str] = None,
        eval_dataset_fraction: float = None,
        comparison_adapter_id: str = None,
        job_id: str = None,
        run_number: int = None,
        client: FineTuningStudioClient = None):

    # TODO: remove hard-coded dependencies on GPU driver for evals
    if run_number != 0:
        time.sleep(20)
    device = "cuda"
    num_gpu_devices = torch.cuda.device_count()
    dataloader = Dataloader()
    logger = ModelLogger(job_id)
    evaluator = ModelEvaluator()

    # Get the model and adapter metadata.
    # given that this is a script that runs on a remote worker (not the same host
    # as the application), need to make gRPC calls to the app server.
    base_model: ModelMetadata = client.GetModel(GetModelRequest(id=base_model_id)).model
    prompt: PromptMetadata = client.GetPrompt(GetPromptRequest(id=prompt_id)).prompt
    # Load in the generation config and bnb config.
    bnb_config_dict = json.loads(client.GetConfig(GetConfigRequest(
        id=bnb_config_id)).config.config) if bnb_config_id else None
    generation_config_dict = json.loads(client.GetConfig(GetConfigRequest(
        id=generation_config_id)).config.config) if generation_config_id else None

    # Load in the adapter, if present
    if adapter_id is not None:
        adapter: AdapterMetadata = client.GetAdapter(GetAdapterRequest(id=adapter_id)).adapter
        assert adapter.type == AdapterType.PROJECT  # Currently can only evaluate on project adapters
    else:
        adapter = None

    # Load dataset split based on adapters that have already been trained
    if comparison_adapter_id is not None:
        dataset_split: GetDatasetSplitByAdapterMetadata = client.GetDatasetSplitByAdapter(
            GetDatasetSplitByAdapterRequest(adapter_id=comparison_adapter_id)).response
    else:
        # as this is only base model evaluation, no need to do any splitting as all data is unseen
        dataset_split = GetDatasetSplitByAdapterMetadata(
            dataset_fraction=0.2, train_test_split=0.2)  # make them variables

    eval_dataset, eval_column_name = dataloader.fetch_evaluation_dataset(
        dataset_id, client=client, prompt_metadata=prompt, dataset_split=dataset_split, selected_features=selected_features, eval_dataset_fraction=eval_dataset_fraction)
    # Load Model Pipeline
    # TODO: remove dependencies on model and adapter type. Right now this assumes that an adapter
    # is available in the project files location, and that the base model is available
    # on huggingface.
    assert base_model.type == ModelType.HUGGINGFACE

    model_info = None
    if num_gpu_devices == 1:
        pipeline = fetch_pipeline(
            base_model.huggingface_model_name,
            adapter.location if adapter is not None else None,
            device=device,
            bnb_config_dict=bnb_config_dict,
            gen_config_dict=generation_config_dict
        )

        # Log model to MLFlow
        model_info = logger.log_model_pipeline(
            pipeline,
            base_model.huggingface_model_name,
            generation_config_dict,
            adapter.name if adapter is not None else None)
    elif num_gpu_devices > 1:
        mlt = MLFlowTransformers()
        try:
            # Inside try cache to avoid failures with wrong adapters
            peft_model, tokenizer = mlt.get_peft_model_and_tokenizer(
                base_model.huggingface_model_name, adapter.location, bnb_config_dict)
            model_info = logger.log_model_multi_gpu(
                peft_model,
                tokenizer,
                generation_config_dict,
                base_model.huggingface_model_name, adapter.name if adapter is not None else None)
        except BaseException:
            # need to improve logic for this. Not sure if this is desired behavior
            raise ValueError("Failed to load peft model. Can run eval on only base model.")
            # commenting the below lines untill we get a  none passed in the adapter field
            # base_model, tokenizer = mlt.get_base_model_and_tokenizer(base_model.huggingface_model_name, bnb_config_dict)
            # model_info = logger.log_model_multi_gpu(base_model, tokenizer)
    else:
        raise ValueError("The driver script is currently set up to handle only GPU evaluation.")

    # Evaluate model
    necessary_eval_dataset = eval_dataset.loc[:, [EVAL_INPUT_COLUMN, EVAL_OUTPUT_COLUM]]
    results = evaluator.evaluate_model(model_info, necessary_eval_dataset, eval_column_name)

    results_df = table_fetcher(results=results)  # pd.DataFrame(results.tables['eval_results_table'])
    merged_results_df = eval_dataset.merge(results_df, on=[EVAL_INPUT_COLUMN, EVAL_OUTPUT_COLUM], how='inner')
    response = EvaluationResponse(metrics=results.metrics, csv=merged_results_df)
    return response


if __name__ == "__main__":
    # Example usage
    driver(dataset_id="a674cd4a-cbcf-490b-ba21-8db2ef689edd",
           base_model_id="f1c3d635-980d-4114-a43e-b3a2eba13910",
           adapter_id="3fe6f50-5be2-4960-bc7b-612843cdf5bd",
           bnb_config_id="538b9d54-6812-4b44-afad-9cb0bd3844ac",
           generation_config_id="aa14660d-d89a-43c0-b323-b09af2f97487")
