from ft.api import *
from datasets import load_dataset_builder
from uuid import uuid4
from cmlapi import CMLServiceApi
from ft.db.dao import FineTuningStudioDao
from ft.db.model import Dataset, Prompt
from ft.utils import require_proto_field
from sqlalchemy import delete
from typing import List
import json
import csv
import datasets


def list_datasets(
    request: ListDatasetsRequest,
    cml: CMLServiceApi = None,
    dao: FineTuningStudioDao = None
) -> ListDatasetsResponse:
    """
    Lists all datasets. In the future, the dataset request object may
    contain information about filtering out datasets.
    """

    # Create a dataset ORM object.
    # query the datasets table using the object.
    # fetch all as a list.
    with dao.get_session() as session:
        result: List[Dataset] = session.query(Dataset).all()

        # Convert to a response object.
        datasets: List[DatasetMetadata] = list(map(lambda mod: mod.to_protobuf(DatasetMetadata), result))

    return ListDatasetsResponse(
        datasets=datasets
    )


def get_dataset(
    request: GetDatasetRequest,
    cml: CMLServiceApi = None,
    dao: FineTuningStudioDao = None
) -> GetDatasetResponse:
    """
    Get a dataset given a dataset request type. Currently datasets can
    only be extracted by an ID.
    """

    with dao.get_session() as session:
        dataset: Dataset = session.query(Dataset).where(Dataset.id == request.id).one()
        dataset_md: DatasetMetadata = dataset.to_protobuf(DatasetMetadata)

    return GetDatasetResponse(
        dataset=dataset_md
    )


def _validate_non_empty_field(dataset_name: str) -> None:
    """
    Validate that an input string is more than just blank space.
    """

    # Ensure the huggingface_name is not an empty string after stripping out spaces
    dataset_name = dataset_name.strip()
    if not dataset_name:
        raise ValueError("Dataset name cannot be an empty string or only spaces.")


def _validate_dataset_ends_with(dataset_name: str, ends_with: str) -> None:
    """
    Validate that a dataset string ends with a certain suffix.
    """
    if not dataset_name.endswith(ends_with):
        raise ValueError(f"Dataset {dataset_name} does not end with {ends_with}")


def _validate_huggingface_dataset_request(request: AddDatasetRequest, dao: FineTuningStudioDao) -> None:
    """
    Run all validations on a huggingface dataset request.
    """

    require_proto_field(request, 'huggingface_name')
    dataset_name = request.huggingface_name
    _validate_non_empty_field(dataset_name)

    # Check if the dataset already exists
    with dao.get_session() as session:
        existing_datasets: List[Dataset] = session.query(Dataset).all()
        if any(ds.huggingface_name == dataset_name for ds in existing_datasets):
            raise ValueError(f"Dataset with name '{dataset_name}' of type {request.type} already exists.")


def _validate_local_dataset_request(request: AddDatasetRequest, dao: FineTuningStudioDao) -> None:
    """
    Run all validations on a local dataset request.
    """

    require_proto_field(request, 'location')
    dataset_name = request.location
    _validate_non_empty_field(dataset_name)

    # Check if the dataset already exists
    with dao.get_session() as session:
        existing_datasets: List[Dataset] = session.query(Dataset).all()
        if any(ds.location == dataset_name for ds in existing_datasets):
            raise ValueError(f"Dataset with name '{dataset_name}' of type {request.type} already exists.")


def _validate_local_csv_dataset_request(request: AddDatasetRequest, dao: FineTuningStudioDao) -> None:
    """
    Validate all fields on a local CSV dataset request.
    """

    require_proto_field(request, 'location')
    require_proto_field(request, 'name')
    _validate_non_empty_field(request.location)
    _validate_dataset_ends_with(request.location, '.csv')
    _validate_non_empty_field(request.name)
    dataset_name = request.name

    # Check if the dataset already exists
    with dao.get_session() as session:
        existing_datasets: List[Dataset] = session.query(Dataset).all()
        if any(ds.name == dataset_name for ds in existing_datasets):
            raise ValueError(f"Dataset with name '{dataset_name}' of type {request.type} already exists.")


def _validate_add_dataset_request(request: AddDatasetRequest, dao: FineTuningStudioDao) -> None:
    # Check for required fields in AddDatasetRequest
    require_proto_field(request, 'type')

    if request.type == DatasetType.HUGGINGFACE:
        _validate_huggingface_dataset_request(request, dao)
    elif request.type == DatasetType.PROJECT:
        _validate_local_dataset_request(request, dao)
    elif request.type == DatasetType.PROJECT_CSV:
        _validate_local_csv_dataset_request(request, dao)


def extract_features_from_csv(location: str) -> List[str]:
    """
    Extract dataset features from a dataset. Studio requires
    the first row to be the dataset "features", which are just
    the names of the columns.
    """

    # Load in the dataset to extract features
    with open(location, mode='r') as file:
        reader = csv.reader(file)
        features = next(reader)  # Get the header row
    return features


def add_dataset(request: AddDatasetRequest, cml: CMLServiceApi = None,
                dao: FineTuningStudioDao = None) -> AddDatasetResponse:
    """
    Retrieve dataset information without fully loading it into memory.
    """
    # Validate the AddDatasetRequest
    _validate_add_dataset_request(request, dao)

    response = AddDatasetResponse()

    # Create a new dataset metadata for the imported dataset.
    if request.type == DatasetType.HUGGINGFACE:
        try:
            # Get dataset information without loading it into memory.
            dataset_builder = load_dataset_builder(request.huggingface_name.strip())
            dataset_info = dataset_builder.info

            # Extract features from the dataset info.
            features = list(dataset_info.features.keys())

            # Add the dataset to the database.
            with dao.get_session() as session:
                dataset = Dataset(
                    id=str(uuid4()),
                    type=request.type,
                    features=json.dumps(features),
                    name=request.huggingface_name.strip(),
                    huggingface_name=request.huggingface_name.strip(),
                    description=dataset_info.description
                )
                session.add(dataset)

                metadata: DatasetMetadata = dataset.to_protobuf(DatasetMetadata)
                response = AddDatasetResponse(dataset=metadata)

        except Exception as e:
            raise ValueError(f"Failed to add dataset. {e}")
    elif request.type == DatasetType.PROJECT_CSV:

        features = extract_features_from_csv(request.location)

        with dao.get_session() as session:
            dataset = Dataset(
                id=str(uuid4()),
                type=request.type,
                features=json.dumps(features),
                name=request.name,
                location=request.location.strip(),
                description=request.location.strip(),  # TODO: add description support
            )
            session.add(dataset)

            metadata: DatasetMetadata = dataset.to_protobuf(DatasetMetadata)
            response = AddDatasetResponse(dataset=metadata)

    else:
        raise ValueError(f"Dataset type [{request.type}] is not yet supported.")

    return response


def remove_dataset(
    request: RemoveDatasetRequest,
    cml: CMLServiceApi = None,
    dao: FineTuningStudioDao = None
) -> RemoveDatasetResponse:

    with dao.get_session() as session:
        session.execute(delete(Dataset).where(Dataset.id == request.id))
        if request.remove_prompts:
            session.execute(delete(Prompt).where(Prompt.dataset_id == request.id))

    return RemoveDatasetResponse()


def load_dataset_into_memory(dataset: DatasetMetadata) -> datasets.DatasetDict:
    """
    Load a dataset into memory based on the type of dataset. Assumes that the
    dataset metadata message has already been validated.

    Some datasets are stored in HF as a Dataset, and others as a DatasetDict. This method
    will guarantee that a DatasetDict is returned, and if a Dataset is stored as just a
    Dataset on Huggingface, this will return a DatasetDict with just one 'train' mapping.

    TODO: We need a cleaner separation of server-side logic (for example, the majory
    of the methods in this file) versus helper python utils that should be available
    to a user. Potentially we should be moving our server impl logic into an srv/ or something
    similar, and then have util functions at the upper level here. This will be addressed
    in a future sprint.
    """

    if dataset.type == DatasetType.HUGGINGFACE:
        ds = datasets.load_dataset(dataset.huggingface_name)
    elif dataset.type == DatasetType.PROJECT_CSV:
        ds = datasets.load_dataset('csv', data_files=dataset.location)
    else:
        raise ValueError(f"Dataset type '{dataset.type}' not supported for this method.")

    # Ensure DatasetDict is being returned.
    if isinstance(ds, dict) or isinstance(ds, datasets.DatasetDict):
        dataset_dict = ds
    elif isinstance(ds, datasets.Dataset):
        dataset_dict = datasets.DatasetDict({'train': ds})
    else:
        raise ValueError(f"Dataset {dataset.name} loading into memory with unsupported type: '{type(dataset)}'")

    return dataset_dict
