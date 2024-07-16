from abc import ABC, abstractmethod
from ft.dataset import DatasetMetadata, ImportDatasetRequest, DatasetType, ImportDatasetResponse
from typing import List 
from datasets import load_dataset
from ft.state import get_state
from uuid import uuid4


class DatasetsManagerBase(ABC):
    """
    Base class for a dataset adapter. A custom implementation
    can be written and the application logic will not change.
    """

    def __init__(self):
        return
    
    @abstractmethod
    def list_datasets(self) -> List[DatasetMetadata]:
        pass

    @abstractmethod
    def get_dataset(self, id: str) -> DatasetMetadata:
        pass

    @abstractmethod
    def import_dataset(self, request: ImportDatasetRequest) -> ImportDatasetResponse:
        pass



class DatasetsManagerSimple(DatasetsManagerBase):

    def list_datasets(self) -> List[DatasetMetadata]:
        return get_state().datasets

    def get_dataset(self, id: str) -> DatasetMetadata:
        """
        Return dataset metadata by dataset ID.
        """
        datasets = get_state().datasets
        datasets = list(filter(lambda x: x.id == id, datasets))
        assert len(datasets) == 1
        return datasets[0]

    def import_dataset(self, request: ImportDatasetRequest) -> ImportDatasetResponse:
        """
        Load a dataset into the App.
        """

        # Create a new dataset metadata for the imported dataset.
        metadata = DatasetMetadata(
            id=str(uuid4()),
            type=request.type,
            name=None,
            description=None,
            huggingface_name=None,
            location=None,
            features=None
        )

        if request.type == DatasetType.HUGGINGFACE:
            
            # Load the dataset into memory.
            loaded_dataset = load_dataset(request.huggingface_name)

            # Lots of datasets have a train split for fine tuning. We will
            # explicitly assume this is the case for the scope of this AMP.
            if "train" in loaded_dataset:
                loaded_dataset = loaded_dataset["train"]

            # Extract out the fields.
            features = loaded_dataset.column_names
            metadata.features = features    
            metadata.huggingface_name = request.huggingface_name
            metadata.name = request.huggingface_name

        else:
            raise ValueError(f"Dataset type [{request.type}] is not yet supported.")
        
        return ImportDatasetResponse(
            dataset=metadata
        )