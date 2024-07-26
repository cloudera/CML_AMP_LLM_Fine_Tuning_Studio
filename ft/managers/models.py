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
            return ImportModelResponse(
                model=ModelMetadata(
                    id=str(uuid4()),
                    type=ModelType.HUGGINGFACE,
                    name=request.huggingface_name,
                    huggingface_model_name=request.huggingface_name
                )
            )
        else:
            print("ERROR: cannot import model of this type.")
            return ImportModelResponse()
        

    def export_model(self, request: ExportModelRequest) -> ExportModelResponse:
        pass
    