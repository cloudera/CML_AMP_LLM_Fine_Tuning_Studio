from enum import Enum


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
