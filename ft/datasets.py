from ft.api import *
from datasets import load_dataset_builder
from uuid import uuid4
from cmlapi import CMLServiceApi
from ft.db.dao import FineTuningStudioDao
from ft.db.model import Dataset, Prompt
from sqlalchemy import delete
from typing import List
import json


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


def add_dataset(request: AddDatasetRequest, cml: CMLServiceApi = None,
                dao: FineTuningStudioDao = None) -> AddDatasetResponse:
    """
    Retrieve dataset information without fully loading it into memory.
    """
    response = AddDatasetResponse()

    # Create a new dataset metadata for the imported dataset.
    if request.type == DatasetType.HUGGINGFACE:
        try:
            # Check if the dataset already exists

            with dao.get_session() as session:
                existing_datasets: List[Dataset] = session.query(Dataset).all()

                if any(ds.huggingface_name == request.huggingface_name for ds in existing_datasets):
                    raise ValueError(f"Dataset with name '{request.huggingface_name}' already exists.")

            # Get dataset information without loading it into memory.
            dataset_builder = load_dataset_builder(request.huggingface_name)
            dataset_info = dataset_builder.info

            # Extract features from the dataset info.
            features = list(dataset_info.features.keys())

            # Add the datasets.
            with dao.get_session() as session:
                dataset = Dataset(
                    id=str(uuid4()),
                    type=request.type,
                    features=json.dumps(features),
                    name=request.huggingface_name,
                    huggingface_name=request.huggingface_name,
                    description=dataset_info.description
                )
                session.add(dataset)
                session.commit()

                metadata: DatasetMetadata = dataset.to_protobuf(DatasetMetadata)
                response = AddDatasetResponse(dataset=metadata)

        except Exception as e:
            raise ValueError(f"Failed to load dataset. {e}")

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
        session.commit()

    return RemoveDatasetResponse()
