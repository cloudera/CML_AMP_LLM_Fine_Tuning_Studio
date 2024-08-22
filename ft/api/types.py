from enum import Enum


class DatasetType(str, Enum):
    """
    Type of dataset that runs.
    """
    HUGGINGFACE = 'huggingface'
    PROJECT = 'project'
