from pydantic import BaseModel
from enum import Enum
from typing import Optional
from ft.cml import RegisteredModelMetadata


class ModelType(Enum):
    """
    Type of model. This AMP currently supports
    loading models in huggingface, from CML's
    model registry, and finally from local project
    files within a CML workspace.
    """

    HUGGINGFACE = "huggingface"
    """
    Huggingface model.
    """

    PROJECT = "project"
    """
    Model imported from project files.

    TODO: determine a way to extract model framework from the content
    of the provided file directory (or by other parameters
    in the model metadata request)
    """

    MODEL_REGISTRY = "model_registry"
    """
    Model was imported from CML Model Registry.
    """


class ModelFrameworkType(Enum):
    """
    The model framework used for this model. Depending on
    the model type (i.e. HF, project, registry), handling
    the model may be different (for example, for local projects,
    we should specify a file/packaging format for ONNX models.)

    TODO: for the most part, this AMP only supports pytorch models
    at this time. we should support other frameworks.
    """
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    ONNX = "onnx"


class ExportType(Enum):
    """
    Type of model export to be performed. Based on
    the type of export occurring,
    """

    PROJECT = "project"
    """
    Export a model to somewhere in project files.

    TODO: add support to export models to project directory.
    """

    MODEL_REGISTRY = "model_registry"
    """
    Export a model to the model registry.
    """


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

    framework: Optional[ModelFrameworkType] = None
    """
    Optional model framework type.
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

    registered_model: Optional[RegisteredModelMetadata] = None
    """
    Metadata on the registered model with CML model registry,
    if this model is a model registry type.
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

    model_registry_id: Optional[str] = None
    """
    Model ID of the model in the model registry of the workspace.
    Used when importing models from model registries.
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

    model: Optional[ModelMetadata] = None
    """
    Registered model artifact with CML Model Registry
    """
