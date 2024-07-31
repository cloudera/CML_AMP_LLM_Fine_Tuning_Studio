from abc import ABC, abstractmethod
from ft.model import (
    ModelMetadata,
    ImportModelRequest,
    ImportModelResponse,
    ModelType,
    ExportModelRequest,
    ExportModelResponse
)
from typing import List
from ft.state import get_state
from uuid import uuid4
from huggingface_hub import HfApi

class ModelsManagerBase(ABC):
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

class ModelsManagerSimple(ModelsManagerBase):
    def list_models(self) -> List[ModelMetadata]:
        return get_state().models

    def import_model(self, request: ImportModelRequest) -> ImportModelResponse:
        if request.type == ModelType.HUGGINGFACE:
            try:
                # Check if the model already exists
                existing_models = get_state().models
                if any(model.huggingface_model_name == request.huggingface_name for model in existing_models):
                    raise ValueError(f"Model with name '{request.huggingface_name}' already exists.")
                
                # Use HfApi to check if the model exists
                api = HfApi()
                model_info = api.model_info(request.huggingface_name)

                # Create model metadata for the imported model
                model_metadata = ModelMetadata(
                    id=str(uuid4()),
                    type=ModelType.HUGGINGFACE,
                    name=request.huggingface_name,
                    huggingface_model_name=request.huggingface_name,
                )

                return ImportModelResponse(model=model_metadata)
            except Exception as e:
                raise ValueError(f"ERROR: Failed to load Hugging Face model. {e}")
        else:
            raise ValueError("ERROR: Cannot import model of this type.")
    
    def export_model(self, request: ExportModelRequest) -> ExportModelResponse:
        pass
