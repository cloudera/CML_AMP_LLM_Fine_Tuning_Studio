from ft.api import *
from uuid import uuid4
from huggingface_hub import HfApi
from huggingface_hub.hf_api import ModelInfo
from cmlapi import RegisteredModel, RegisteredModelVersion, ModelVersionMetadata, MLflowMetadata, CMLServiceApi
from ft.state import write_state
from ft.pipeline import fetch_pipeline
import mlflow
from transformers import GenerationConfig


def list_models(state: AppState, request: ListModelsRequest, cml: CMLServiceApi = None) -> ListModelsResponse:
    """
    List all models. In the future, the request object may
    have filtering routines, which is why this is abstracted out.
    """

    return ListModelsResponse(
        models=state.models
    )


def get_model(state: AppState, request: GetModelRequest, cml: CMLServiceApi = None) -> GetModelResponse:
    """
    Get a model given a model id. Currently models can
    only be extracted by an ID.
    """

    models = state.models
    models = list(filter(lambda x: x.id == request.id, models))
    assert len(models) == 1
    return GetModelResponse(
        model=models[0]
    )


def add_model(state: AppState, request: AddModelRequest, cml: CMLServiceApi = None) -> AddModelResponse:
    response: AddModelResponse = AddModelResponse()

    if request.type == ModelType.MODEL_TYPE_HUGGINGFACE:
        try:
            # Check if the model already exists
            existing_models = state.models
            if any(model.huggingface_model_name == request.huggingface_name for model in existing_models):
                raise ValueError(f"Model with name '{request.huggingface_name}' already exists.")

            # Use HfApi to check if the model exists
            api = HfApi()
            model_info: ModelInfo = api.model_info(request.huggingface_name)

            # Create model metadata for the imported model
            model_metadata = ModelMetadata(
                id=str(uuid4()),
                type=ModelType.MODEL_TYPE_HUGGINGFACE,
                name=request.huggingface_name,
                huggingface_model_name=request.huggingface_name,
            )

            response = AddModelResponse(
                model=model_metadata
            )
        except Exception as e:
            raise ValueError(f"ERROR: Failed to load Hugging Face model. {e}")
    elif request.type == ModelType.MODEL_TYPE_MODEL_REGISTRY:

        assert request.model_registry_id is not None

        # Get the model registry metadata.
        try:
            registered_model: RegisteredModel = cml.get_registered_model(request.model_registry_id)

            # TODO: Support multiple model registry model versions.
            first_version: RegisteredModelVersion = registered_model.model_versions[0]
            model_version_metadata: ModelVersionMetadata = first_version.model_version_metadata
            mlflow_metadata: MLflowMetadata = model_version_metadata.mlflow_metadata

            model_metadata = ModelMetadata(
                id=str(uuid4()),
                type=ModelType.MODEL_TYPE_MODEL_REGISTRY,
                name=registered_model.name,
                registered_model=RegisteredModelMetadata(
                    cml_registered_model_id=registered_model.model_id,
                    mlflow_experiment_id=mlflow_metadata.experiment_id,
                    mlflow_run_id=mlflow_metadata.run_id,
                )
            )

            response = AddModelResponse(
                model=model_metadata
            )
        except Exception as e:
            raise ValueError(f"ERROR: Failed to load model registry model. {e}")

    else:
        raise ValueError("ERROR: Cannot import model of this type.")

    # Write new model to app state if adding was successful
    if not response == AddModelResponse():
        state.models.append(response.model)
        write_state(state)

    return response


def export_model(state: AppState, request: ExportModelRequest, cml: CMLServiceApi = None) -> ExportModelResponse:
    """
    Export model. Currently only applies to exporting to model registry.
    """

    if not request.type == ModelType.MODEL_TYPE_MODEL_REGISTRY:
        raise ValueError("Model exports are only supported to model registry at this time.")

    response: ExportModelResponse = ExportModelResponse()

    # TODO: extract out individual model filtering so we don't have to
    # call the full request/response envelope here.
    model: ModelMetadata = list(filter(lambda x: x.id == request.model_id, state.models))[0]
    adapter: AdapterMetadata = list(filter(lambda x: x.id == request.adapter_id, state.adapters))[0]

    # For now, let's assume HF model is available. If not, we should be ideally
    # raising an error or handling custom models differently.
    adapter_location_or_name = adapter.location if adapter.type == AdapterType.ADAPTER_TYPE_PROJECT else adapter.huggingface_name
    pipeline = fetch_pipeline(
        model_name=model.huggingface_model_name,
        adapter_name=adapter_location_or_name)

    signature = mlflow.models.infer_signature(
        model_input="What are the three primary colors?",
        model_output="The three primary colors are red, yellow, and blue.",
    )

    # TODO: pull out generation config to arguments
    config = GenerationConfig(
        do_sample=True,
        temperature=0.8,
        max_new_tokens=60,
        top_p=1
    )

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=pipeline,
            torch_dtype='float16',
            artifact_path="custom-pipe",        # artifact_path can be dynamic
            signature=signature,
            registered_model_name=request.model_name,  # model_name can be dynamic
            model_config=config.to_dict()
        )

    # We aren't doing any error handling at this time, and we aren't
    # explicitly using the return metadata yet.
    return response


def remove_model(state: AppState, request: RemoveModelRequest, cml: CMLServiceApi = None) -> RemoveModelResponse:
    models = list(filter(lambda x: not x.id == request.id, state.models))
    write_state(AppState(
        datasets=state.datasets,
        prompts=state.prompts,
        adapters=state.adapters,
        fine_tuning_jobs=state.fine_tuning_jobs,
        evaluation_jobs=state.evaluation_jobs,
        models=models
    ))
    return RemoveModelResponse()
