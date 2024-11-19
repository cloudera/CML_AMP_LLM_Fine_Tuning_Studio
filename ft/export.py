from uuid import uuid4
import cmlapi
from cmlapi import CMLServiceApi
from ft.api import *
import os
import json
from transformers import GenerationConfig
import mlflow

from ft.db.dao import FineTuningStudioDao
from ft.db.model import Adapter, Model
from ft.pipeline import fetch_pipeline
from ft.consts import CML_MODEL_PREDICT_SCRIPT_FILEPATH, DEFAULT_GENERATIONAL_CONFIG


def get_cml_model_inference_runtime_identifier(cml: CMLServiceApi) -> str:
    """
    Get a runtime ID to be used for inference in CML models. For now, we will use
    the same runtime ID as the fine tuning training jobs, since that runtime ID has
    all components necessary for GPU inference on CUDA enabled devices.
    """
    project_id = os.getenv("CDSW_PROJECT_ID")
    base_job_name = "Finetuning_Base_Job"

    ft_base_job_id = cml.list_jobs(project_id,
                                   search_filter='{"name":"%s"}' % base_job_name).jobs[0].id
    template_job = cml.get_job(
        project_id=project_id,
        job_id=ft_base_job_id
    )
    return template_job.runtime_identifier


def _validate_export_model_request(request: ExportModelRequest, dao: FineTuningStudioDao) -> None:
    # Check for required fields in StartEvaluationJobRequest
    required_fields = [
        "type", "base_model_id", "adapter_id", "model_name"
    ]

    for field in required_fields:
        if not getattr(request, field):
            raise ValueError(f"Field '{field}' is required in StartEvaluationJobRequest.")

    # Ensure certain string fields are not empty after stripping out spaces
    string_fields = [
        "type", "base_model_id", "adapter_id", "model_name"
    ]

    for field in string_fields:
        try:
            field_value = getattr(request, field).strip()
        except BaseException:
            # repeatable proto strip error
            field_value = getattr(request, field)
        if not field_value:
            raise ValueError(f"Field '{field}' cannot be an empty string or only spaces.")

    # Check if the referenced base_model_id exists in the database
    with dao.get_session() as session:
        base_model_id = request.base_model_id
        adapter_id = request.adapter_id
        if not session.query(Model).filter_by(id=base_model_id.strip()).first():
            raise ValueError(f"Model with ID '{base_model_id}' does not exist.")
        # Check if the referenced adapter_id exists in the database or is the default base model
        if not session.query(Adapter).filter_by(id=adapter_id.strip()).first():
            raise ValueError(f"Adapter with ID '{adapter_id}' does not exist.")


def export_model_registry_model(request: ExportModelRequest, cml: CMLServiceApi = None,
                                dao: FineTuningStudioDao = None) -> ExportModelResponse:
    """
    Export a model to CML model registry. For now, model registry cannot deploy huggingface models and
    adapters to Cloudera AI Inference, so exporting a model to model registry doesn't enable
    more features for the model.
    """

    _validate_export_model_request(request, dao)

    # Right now, our response object contains a model metadata nullable object, just in case
    # we want to automatically add other model types to the studio in the future.
    response: ExportModelResponse = ExportModelResponse()

    pipeline = None
    if not getattr(request, "generation_config"):
        config = GenerationConfig(**DEFAULT_GENERATIONAL_CONFIG)
    else:
        gen_config = json.loads(request.generation_config)
        config = GenerationConfig(**gen_config)
    with dao.get_session() as session:
        model: Model = session.query(Model).where(Model.id == request.base_model_id).one()
        assert model.type == ModelType.HUGGINGFACE

        # For now, require adapter.
        adapter: Adapter = session.query(Adapter).where(Adapter.id == request.adapter_id).one()
        adapter_location_or_name = adapter.location if adapter.type == AdapterType.PROJECT else adapter.huggingface_name

        pipeline = fetch_pipeline(
            model_name=model.huggingface_model_name,
            adapter_name=adapter_location_or_name,
            gen_config_dict=config.to_dict())

    signature = mlflow.models.infer_signature(
        model_input="What are the three primary colors?",
        model_output="The three primary colors are red, yellow, and blue.",
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


def deploy_cml_model(request: ExportModelRequest,
                     cml: CMLServiceApi = None, dao: FineTuningStudioDao = None) -> ExportModelResponse:
    """
    Launch a cml export async process
    """
    _validate_export_model_request(request, dao)

    response = ExportModelResponse()
    job_id = str(uuid4())

    # CML model export requires a HF model and a project-specific adapter.
    base_model_hf_name = None
    adapter_location = None
    gen_config_str = json.dumps(DEFAULT_GENERATIONAL_CONFIG)
    if getattr(request, "generation_config"):
        gen_config_str = request.generation_config

    with dao.get_session() as session:
        model: Model = session.query(Model).where(Model.id == request.base_model_id).one()
        adapter: Adapter = session.query(Adapter).where(Adapter.id == request.adapter_id).one()
        assert model.type == ModelType.HUGGINGFACE
        assert adapter.type == AdapterType.PROJECT
        base_model_hf_name = model.huggingface_model_name
        adapter_location = adapter.location

    print(f"Deploying {base_model_hf_name} + {adapter_location} as a CML model...")
    project_id = os.getenv("CDSW_PROJECT_ID")

    client = cmlapi.default_client()
    project: cmlapi.Project = client.get_project(project_id)
    model_body = cmlapi.CreateModelRequest(project_id=project.id,
                                           name=request.model_name,
                                           description=request.model_description)
    model = client.create_model(model_body, project.id)
    short_model_deployment = cmlapi.ShortCreateModelDeployment(
        cpu=2,
        memory=8,
        nvidia_gpus=1,
        environment={
            "FINE_TUNING_STUDIO_BASE_MODEL_HF_NAME": base_model_hf_name,
            "FINE_TUNING_STUDIO_ADAPTER_LOCATION": adapter_location,
            "FINE_TUNING_STUDIO_GEN_CONFIG_STRING": gen_config_str
        }
    )
    model_build_body = cmlapi.CreateModelBuildRequest(
        project_id=project.id,
        model_id=model.id,
        file_path=CML_MODEL_PREDICT_SCRIPT_FILEPATH,
        function_name="api_wrapper",
        runtime_identifier=get_cml_model_inference_runtime_identifier(client),
        kernel="python3",
        auto_deployment_config=short_model_deployment,
        auto_deploy_model=True)
    print(model_build_body)

    model_build_thread = client.create_model_build(model_build_body, project.id, model.id)
    print(f"model build thread created: {model_build_thread}")

    response = ExportModelResponse(
        base_model_id=request.base_model_id,
        adapter_id=request.adapter_id
    )

    return response
