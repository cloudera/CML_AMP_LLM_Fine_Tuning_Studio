from enum import Enum
from pydantic import BaseModel
from typing import Dict, Any


class DatasetType(str, Enum):
    """
    Type of dataset that runs.
    """
    HUGGINGFACE = 'huggingface'
    PROJECT = 'project'


class ConfigType(str, Enum):
    """
    Type of configuration in the config store.
    """
    TRAINING_ARGUMENTS = "training_arguments"
    BITSANDBYTES_CONFIG = "bitsandbytes_config"
    GENERATION_CONFIG = "generation_config"
    LORA_CONFIG = "lora_config"
    CUSTOM = "custom"
    AXOLOTL = "axolotl"
    AXOLOTL_DATASET_FORMATS = "axolotl_dataset_formats"


class FineTuningFrameworkType(str, Enum):
    """
    Type of fine tuning framework being deployed.
    """
    LEGACY = "legacy"
    AXOLOTL = "axolotl"


class AdapterType(str, Enum):
    """
    Type of adapter in the project
    """
    PROJECT = "project"
    HUGGINGFACE = "huggingface"
    MODEL_REGISTRY = "model_registry"


class PromptType(str, Enum):
    IN_PLACE = "in_place"


class ModelType(str, Enum):
    HUGGINGFACE = "huggingface"
    PROJECT = "project"
    MODEL_REGISTRY = "model_registry"


class ModelFrameworkType(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"


class EvaluationJobType(str, Enum):
    MFLOW = "mlflow"


class DatasetFormatInfo(BaseModel):
    name: str
    description: str
    format: Dict[str, Any]  # Allows any structure within the format dictionary


class DatasetFormatsCollection(BaseModel):
    dataset_formats: Dict[str, DatasetFormatInfo]
