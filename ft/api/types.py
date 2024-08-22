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
