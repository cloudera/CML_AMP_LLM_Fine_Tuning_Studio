from ft.api import *
from uuid import uuid4
from huggingface_hub import HfApi
from huggingface_hub.hf_api import ModelInfo
from cmlapi import RegisteredModel, RegisteredModelVersion, ModelVersionMetadata, MLflowMetadata, CMLServiceApi

from typing import List
from sqlalchemy import delete

from ft.db.dao import FineTuningStudioDao
from ft.db.model import Model
from ft.export import deploy_cml_model, export_model_registry_model


def list_models(request: ListModelsRequest, cml: CMLServiceApi = None,
                dao: FineTuningStudioDao = None) -> ListModelsResponse:
    """
    List all models. In the future, the request object may
    have filtering routines, which is why this is abstracted out.
    """
    with dao.get_session() as session:
        models: List[Model] = session.query(Model).all()
        return ListModelsResponse(
            models=list(map(
                lambda x: x.to_protobuf(ModelMetadata),
                models
            ))
        )


def get_model(request: GetModelRequest, cml: CMLServiceApi = None, dao: FineTuningStudioDao = None) -> GetModelResponse:
    """
    Get a model given a model id. Currently models can
    only be extracted by an ID.
    """
    with dao.get_session() as session:
        return GetModelResponse(
            model=session
            .query(Model)
            .where(Model.id == request.id)
            .one()
            .to_protobuf(ModelMetadata)
        )


def _validate_add_model_request(request: AddModelRequest, dao: FineTuningStudioDao) -> None:
    # Check for required fields in AddModelRequest
    required_fields = ["type"]

    for field in required_fields:
        if not getattr(request, field):
            raise ValueError(f"Field '{field}' is required in AddModelRequest.")

    # Ensure the huggingface_name is not an empty string after stripping out spaces
    huggingface_name = request.huggingface_name.strip()
    if huggingface_name:
        # Check if the model already exists
        with dao.get_session() as session:
            existing_models: List[Model] = session.query(Model).all()
            if any(model.huggingface_model_name == huggingface_name for model in existing_models):
                raise ValueError(f"Model with name '{huggingface_name}' already exists.")
    else:
        raise ValueError("Hugging Face model name cannot be an empty string or only spaces.")


def add_model(request: AddModelRequest, cml: CMLServiceApi = None, dao: FineTuningStudioDao = None) -> AddModelResponse:
    response: AddModelResponse = AddModelResponse()

    # Validate the AddModelRequest
    _validate_add_model_request(request, dao)

    if request.type == ModelType.HUGGINGFACE:
        try:
            with dao.get_session() as session:
                # Use HfApi to check if the model exists
                api = HfApi()
                model_info: ModelInfo = api.model_info(request.huggingface_name.strip())

                model: Model = Model(
                    id=str(uuid4()),
                    type=ModelType.HUGGINGFACE,
                    name=request.huggingface_name.strip(),
                    huggingface_model_name=request.huggingface_name.strip()
                )
                session.add(model)

                response = AddModelResponse(
                    model=model.to_protobuf(ModelMetadata)
                )
        except Exception as e:
            raise ValueError(f"ERROR: Failed to load Hugging Face model. {e}")

    elif request.type == ModelType.MODEL_REGISTRY:

        assert request.model_registry_id is not None

        # Get the model registry metadata.
        try:
            with dao.get_session() as session:

                registered_model: RegisteredModel = cml.get_registered_model(request.model_registry_id)

                # TODO: Support multiple model registry model versions.
                first_version: RegisteredModelVersion = registered_model.model_versions[0]
                model_version_metadata: ModelVersionMetadata = first_version.model_version_metadata
                mlflow_metadata: MLflowMetadata = model_version_metadata.mlflow_metadata

                model: Model = Model(
                    id=str(uuid4()),
                    type=ModelType.MODEL_REGISTRY,
                    name=registered_model.name,
                    cml_registered_model_id=registered_model.model_id,
                    mlflow_experiment_id=mlflow_metadata.experiment_id,
                    mlflow_run_id=mlflow_metadata.run_id,
                )
                session.add(model)

                response = AddModelResponse(
                    model=model.to_protobuf(ModelMetadata)
                )
        except Exception as e:
            raise ValueError(f"ERROR: Failed to load model registry model. {e}")

    else:
        raise ValueError("ERROR: Cannot import model of this type.")

    return response


def export_model(request: ExportModelRequest, cml: CMLServiceApi = None,
                 dao: FineTuningStudioDao = None) -> ExportModelResponse:
    """
    Export model outside of Fine Tuning Studio.
    """

    if request.type == ModelExportType.MODEL_REGISTRY:
        return export_model_registry_model(request, cml, dao)
    elif request.type == ModelExportType.CML_MODEL:
        return deploy_cml_model(request, cml, dao)
    else:
        raise ValueError(f"Model export of type '{request.type}' is not supported.")


def remove_model(request: RemoveModelRequest, cml: CMLServiceApi = None,
                 dao: FineTuningStudioDao = None) -> RemoveModelResponse:
    with dao.get_session() as session:
        session.execute(delete(Model).where(Model.id == request.id))
    return RemoveModelResponse()
