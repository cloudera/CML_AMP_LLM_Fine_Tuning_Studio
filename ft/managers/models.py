from abc import ABC, abstractmethod
from ft.api import *
from typing import List
from ft.state import get_state
from ft.managers.cml import CMLManager
from uuid import uuid4
from huggingface_hub import HfApi
from huggingface_hub.hf_api import ModelInfo
from cmlapi import RegisteredModel, RegisteredModelVersion, ModelVersionMetadata, MLflowMetadata


class ModelsManagerBase(ABC):
    """
    Base models manager class responsible for all logic
    behind importing, exporting, and other operations retaining
    to base models, adapters, etc.
    """

    def __init__(self):
        return

    @abstractmethod
    def list_models(self) -> List[ModelMetadata]:
        pass

    @abstractmethod
    def import_model(self, request: ImportModelRequest) -> ImportModelResponse:
        pass

    @abstractmethod
    def export_model(self, request: ExportModelRequest) -> ExportModelResponse:
        pass


class ModelsManagerSimple(ModelsManagerBase, CMLManager):
    """
    Model manager made available when deploying this application in a
    CML workspace. This is the default model manager activated when deploying
    this code to an AMP.
    """

    def __init__(self):
        CMLManager.__init__(self)

    def list_models(self) -> List[ModelMetadata]:
        return get_state().models

    def import_model(self, request: ImportModelRequest) -> ImportModelResponse:
        response: ImportModelResponse = ImportModelResponse()

        if request.type == ModelType.MODEL_TYPE_HUGGINGFACE:
            try:
                # Check if the model already exists
                existing_models = get_state().models
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

                response = ImportModelResponse(
                    model=model_metadata
                )
            except Exception as e:
                raise ValueError(f"ERROR: Failed to load Hugging Face model. {e}")
        elif request.type == ModelType.MODEL_TYPE_MODEL_REGISTRY:

            assert request.model_registry_id is not None

            # Get the model registry metadata.
            try:
                registered_model: RegisteredModel = self.cml_api_client.get_registered_model(request.model_registry_id)

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

                response = ImportModelResponse(
                    model=model_metadata
                )
            except Exception as e:
                raise ValueError(f"ERROR: Failed to load model registry model. {e}")

        else:
            raise ValueError("ERROR: Cannot import model of this type.")

        return response

    def export_model(self, request: ExportModelRequest) -> ExportModelResponse:
        pass
