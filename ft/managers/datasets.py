from abc import ABC, abstractmethod
from ft.dataset import DatasetMetadata, ImportDatasetRequest, DatasetType, ImportDatasetResponse
from typing import List
from datasets import load_dataset_builder
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
        Retrieve dataset information without fully loading it into memory.
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
            try:
                # Check if the dataset already exists
                existing_datasets = get_state().datasets
                if any(ds.huggingface_name == request.huggingface_name for ds in existing_datasets):
                    raise ValueError(f"Dataset with name '{request.huggingface_name}' already exists.")

                # Get dataset information without loading it into memory.
                dataset_builder = load_dataset_builder(request.huggingface_name)
                dataset_info = dataset_builder.info

                print(dataset_info)

                # Extract features from the dataset info.
                features = list(dataset_info.features.keys())
                metadata.features = features
                metadata.huggingface_name = request.huggingface_name
                metadata.name = request.huggingface_name
                metadata.description = dataset_info.description

            except Exception as e:
                raise ValueError(f"Failed to load dataset. {e}")

        else:
            raise ValueError(f"Dataset type [{request.type}] is not yet supported.")

        return ImportDatasetResponse(
            dataset=metadata
        )
