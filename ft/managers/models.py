from abc import ABC, abstractmethod
from ft.model import ModelMetadata, ImportModelRequest
from typing import List 

class ModelsManagerBase(ABC):
    def __init__(self):
        return

    @abstractmethod
    def list_models() -> List[ModelMetadata]:
        pass 

    @abstractmethod
    def import_model(request: ImportModelRequest):
        pass


class ModelsManagerSimple(ModelsManagerBase):
    def list_models() -> List[ModelMetadata]:
        pass

    def import_model(request: ImportModelRequest):
        return super().import_model()