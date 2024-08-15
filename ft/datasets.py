from ft.api import *
from datasets import load_dataset_builder
from ft.state import write_state, replace_state_field
from uuid import uuid4
from cmlapi import CMLServiceApi


def list_datasets(state: AppState, request: ListDatasetsRequest, cml: CMLServiceApi = None) -> ListDatasetsResponse:
    """
    Lists all datasets. In the future, the dataset request object may
    contain information about filtering out datasets.
    """

    return ListDatasetsResponse(
        datasets=state.datasets
    )


def get_dataset(state: AppState, request: GetDatasetRequest, cml: CMLServiceApi = None) -> GetDatasetResponse:
    """
    Get a dataset given a dataset request type. Currently datasets can
    only be extracted by an ID.
    """

    datasets = state.datasets
    datasets = list(filter(lambda x: x.id == request.id, datasets))
    assert len(datasets) == 1
    return GetDatasetResponse(
        dataset=datasets[0]
    )


def add_dataset(state: AppState, request: AddDatasetRequest, cml: CMLServiceApi = None) -> AddDatasetResponse:
    """
    Retrieve dataset information without fully loading it into memory.
    """
    response = AddDatasetResponse()

    # Create a new dataset metadata for the imported dataset.
    if request.type == DatasetType.DATASET_TYPE_HUGGINGFACE:
        try:
            # Check if the dataset already exists
            existing_datasets = state.datasets
            if any(ds.huggingface_name == request.huggingface_name for ds in existing_datasets):
                raise ValueError(f"Dataset with name '{request.huggingface_name}' already exists.")

            # Get dataset information without loading it into memory.
            dataset_builder = load_dataset_builder(request.huggingface_name)
            dataset_info = dataset_builder.info

            print(dataset_info)

            # Extract features from the dataset info.
            features = list(dataset_info.features.keys())
            metadata = DatasetMetadata(
                id=str(uuid4()),
                type=request.type,
                features=features,
                huggingface_name=request.huggingface_name,
                name=request.huggingface_name,
                description=dataset_info.description
            )
            response = AddDatasetResponse(dataset=metadata)

        except Exception as e:
            raise ValueError(f"Failed to load dataset. {e}")

    else:
        raise ValueError(f"Dataset type [{request.type}] is not yet supported.")

    # If we have a response, add the dataset to the app's state.
    if not response == AddDatasetResponse():
        state.datasets.append(response.dataset)
        write_state(state)

    return response


def remove_dataset(state: AppState, request: RemoveDatasetRequest, cml: CMLServiceApi = None) -> RemoveDatasetResponse:
    """
    TODO: this should be an official request/response type
    """
    datasets = list(filter(lambda x: not x.id == request.id, state.datasets))
    prompts = state.prompts
    if request.remove_prompts:
        prompts = list(filter(lambda x: not x.dataset_id == request.id, state.prompts))

    state = replace_state_field(state, datasets=datasets)
    state = replace_state_field(state, prompts=prompts)
    return RemoveDatasetResponse()
