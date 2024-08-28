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
import torch


def driver(
        dataset_id: str = None,
        base_model_id: str = None,
        adapter_id: str = None,
        bnb_config_id: str = None,
        generation_config_id: str = None,
        prompt_id: str = None,
        client: FineTuningStudioClient = None):

    # TODO: remove hard-coded dependencies on GPU driver for evals
    device = "cuda"
    num_gpu_devices = torch.cuda.device_count()
    dataloader = Dataloader()
    logger = ModelLogger()
    evaluator = ModelEvaluator()


    # Get the model and adapter metadata.
    # given that this is a script that runs on a remote worker (not the same host
    # as the application), need to make gRPC calls to the app server.
    base_model: ModelMetadata = client.GetModel(GetModelRequest(id=base_model_id)).model
    adapter: AdapterMetadata = client.GetAdapter(GetAdapterRequest(id=adapter_id)).adapter
    prompt : PromptMetadata = client.GetPrompt(GetPromptRequest(id=prompt_id)).prompt
    # Load in the generation config and bnb config.
    bnb_config_dict = json.loads(client.GetConfig(GetConfigRequest(
        id=bnb_config_id)).config.config) if bnb_config_id else None
    generation_config_dict = json.loads(client.GetConfig(GetConfigRequest(
        id=generation_config_id)).config.config) if generation_config_id else None

    # Load dataset
    eval_dataset, eval_column_name = dataloader.fetch_evaluation_dataset(dataset_id, client=client, prompt_metadata = prompt)
    # Load Model Pipeline
    # TODO: remove dependencies on model and adapter type. Right now this assumes that an adapter
    # is available in the project files location, and that the base model is available
    # on huggingface.
    assert base_model.type == ModelType.HUGGINGFACE
    assert adapter.type == AdapterType.PROJECT
    model_info = None
    if num_gpu_devices == 1:
        pipeline = fetch_pipeline(
            base_model.huggingface_model_name,
            adapter.location,
            device=device,
            bnb_config_dict=bnb_config_dict,
            gen_config_dict=generation_config_dict
        )

        # Log model to MLFlow
        model_info = logger.log_model_pipeline(pipeline)
    elif num_gpu_devices > 1:
        mlt = MLFlowTransformers()
        try:
            # Inside try cache to avoid failures with wrong adapters
            peft_model, tokenizer = mlt.get_peft_model_and_tokenizer(
                base_model.huggingface_model_name, adapter.location, bnb_config_dict)
            model_info = logger.log_model_multi_gpu(peft_model, tokenizer)
        except BaseException:
            # need to improve logic for this. Not sure if this is desired behavior
            raise ValueError("Failed to load peft model. Can run eval on only base model.")
            # commenting the below lines untill we get a  none passed in the adapter field
            # base_model, tokenizer = mlt.get_base_model_and_tokenizer(base_model.huggingface_model_name, bnb_config_dict)
            # model_info = logger.log_model_multi_gpu(base_model, tokenizer)
    else:
        raise ValueError("The driver script is currently set up to handle only GPU evaluation.")

    # Evaluate model
    results = evaluator.evaluate_model(model_info, eval_dataset, eval_column_name)

    results_df = pd.DataFrame(results.tables['eval_results_table'])
    response = EvaluationResponse(metrics=results.metrics, csv=results_df)
    return response


if __name__ == "__main__":
    # Example usage
    driver(dataset_id="a674cd4a-cbcf-490b-ba21-8db2ef689edd",
           base_model_id="f1c3d635-980d-4114-a43e-b3a2eba13910",
           adapter_id="3fe6f50-5be2-4960-bc7b-612843cdf5bd",
           bnb_config_id="538b9d54-6812-4b44-afad-9cb0bd3844ac",
           generation_config_id="aa14660d-d89a-43c0-b323-b09af2f97487")
