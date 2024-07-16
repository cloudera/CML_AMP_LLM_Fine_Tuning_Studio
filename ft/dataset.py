from pydantic import BaseModel
from enum import Enum 
from typing import Optional, List


class DatasetType(Enum):
    HUGGINGFACE = "huggingface"
    IMPORTED = "imported"


class DatasetMetadata(BaseModel):
    id: str
    """
    Dataset ID.
    """

    type: DatasetType
    """
    Type of dataset. Can either be pulled from 
    huggingface, or can be uploaded into Projects
    folder manually (so long as the dataset is in 
    the correct format)
    """

    name: Optional[str]
    """
    Human-readable name for the dataset.
    """

    description: Optional[str]
    """
    Description of the dataset.
    """

    huggingface_name: Optional[str] 
    """
    Canonical huggingface dataset name (can be used to find
    huggingface hub)
    """

    location: Optional[str]
    """
    Project-relative location of the dataset that is
    loaded into the app's state.
    """

    features: Optional[List[str]]
    """
    List of features in the dataset. As of now, these features
    are automatically loaded in every time there is a new dataset
    that is loaded from huggingface. The reason this is extracted
    out of the dataset and stored in metadata is so we can view
    the features (columns) of the dataset as part of the App state
    without loading the entire dataset into memory.
    """


class ImportDatasetRequest(BaseModel):

    type: DatasetType
    """
    Type of dataset to import.
    """

    huggingface_name: Optional[str] 
    """
    Name of the huggingface dataset. This is the full huggingface
    dataset name used to identify the dataset on HF hub.
    """

    location: Optional[str]
    """
    Project-relative location of the dataset to import.
    """


class ImportDatasetResponse(BaseModel):
    dataset: Optional[DatasetMetadata]
    
