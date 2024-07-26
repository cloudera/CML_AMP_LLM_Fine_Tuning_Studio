from pydantic import BaseModel
from enum import Enum
from typing import Optional


class ModelType(Enum):
    HUGGINGFACE = "huggingface"
    ONNX = "onnx"
    PYTORCH = "pytorch"
    MODEL_REGISTRY = "model_registry"


class ExportType(Enum):
    """
    Type of model export to be performed. Based on
    the type of export occurring,
    """
    ONNX = "onnx"
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


class RegisteredModelMetadata(BaseModel):
    """
    Represents a CML-registered model.
    """

    model_id: str
    """
    Model ID of the registered model
    """

    adapter_id: Optional[str] = None
    """
    Optionally any adapter that is tied to
    this specific registered model
    """

    cml_model_id: str
    """
    CML Model ID of the registered model. This can
    be used with CML API to extract model information.
    """

    url: Optional[str] = None
    """
    URL which can be used to view this registered model
    directly within the CML ecosystem.
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


class ExportModelRequest(BaseModel):

    type: ExportType
    """
    Type of export model operation to perform.
    """

    model_id: str
    """
    Model ID that should be exported
    """

    adapter_id: Optional[str] = None
    """
    Trained adapter that is to also be
    exported (optional). Depending on the model
    export type, any PEFT adapter weights may be
    merged into the base model.
    """

    model_name: Optional[str] = None
    """
    Human-friendly name to give to the exported
    model. Might not be used if only exporting
    model to a file output (for example, ONNX output)
    """

    artifact_location: Optional[str] = None
    """
    Export output artifact location for export types
    that require file-writing to project files.
    """

    model_description: Optional[str] = None
    """
    Optional model description for those model export
    types that allow for descriptions.
    """


class ExportModelResponse(BaseModel):
    """
    Response type for model exports.
    """

    registered_model: Optional[RegisteredModelMetadata] = None
    """
    Registered model artifact with CML Model Registry
    """
