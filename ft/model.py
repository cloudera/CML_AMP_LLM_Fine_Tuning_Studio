from pydantic import BaseModel
from enum import Enum 
from typing import Optional


class ModelType(Enum):
    HUGGINGFACE = "huggingface"
    ONNX = "onnx"
    PYTORCH = "pytorch"
    MODEL_REGISTRY = "model_registry"
    

class ModelMetadata(BaseModel):
    id: str
    """
    Global identifier for models. 
    
    For the purpose of this
    AMP application, during local ML model loading & inference, 
    model IDs are random unique identifiers that have no
    significance within the CML ecosystem. Evenutally when this 
    AMP is integrated with CML model registry, we will ideally
    be able to have a more significant model ID.
    """

    type: ModelType
    """
    Type of model. This type affects the
    """

    name: Optional[str] = None
    """
    Human-friendly name for the model.
    """

    huggingface_model_name: Optional[str] = None
    """
    Name of the huggingface model. This is the human-readable
    model name that can be used to identify a huggingface model
    on HF hub.
    """



class ImportModelRequest(BaseModel):
    type: ModelType
    """
    Type of model. This type affects the
    """

    huggingface_name: Optional[str] = None
    """
    Name of the huggingface model. This is the full huggingface
    model name used to identify the model on HF hub.
    """ 

class ImportModelResponse(BaseModel):
    model: Optional[ModelMetadata] = None