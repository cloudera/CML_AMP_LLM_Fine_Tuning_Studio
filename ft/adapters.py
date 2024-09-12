from ft.api import *

from cmlapi import CMLServiceApi

from uuid import uuid4

from typing import List

from ft.db.dao import FineTuningStudioDao
from ft.db.model import Adapter, FineTuningJob, Prompt, Model
from ft.consts import TRAINING_DEFAULT_DATASET_FRACTION, TRAINING_DEFAULT_TRAIN_TEST_SPLIT
from sqlalchemy import delete
from sqlalchemy.exc import NoResultFound
import os


def list_adapters(request: ListAdaptersRequest, cml: CMLServiceApi = None,
                  dao: FineTuningStudioDao = None) -> ListAdaptersResponse:
    """
    Right now we don't do any filtering in this op, but we might in the future.
    """
    with dao.get_session() as session:
        adapters: List[Adapter] = session.query(Adapter).all()
        return ListAdaptersResponse(
            adapters=list(map(
                lambda x: x.to_protobuf(AdapterMetadata),
                adapters
            ))
        )


def get_adapter(request: GetAdapterRequest, cml: CMLServiceApi = None,
                dao: FineTuningStudioDao = None) -> GetAdapterResponse:

    with dao.get_session() as session:
        return GetAdapterResponse(
            adapter=session
            .query(Adapter)
            .where(Adapter.id == request.id)
            .one()
            .to_protobuf(AdapterMetadata)
        )


def _validate_add_adapter_request(request: AddAdapterRequest, dao: FineTuningStudioDao) -> None:
    """
    Validate the parameters of the AddAdapterRequest.

    This function checks for necessary conditions and constraints in the request
    parameters and raises exceptions if validation fails.
    """

    # Check for required fields in AddAdapterRequest
    required_fields = ["name", "model_id", "location"]

    for field in required_fields:
        if not getattr(request, field):
            raise ValueError(f"Field '{field}' is required in AddAdapterRequest.")

    # Ensure certain string fields are not empty after stripping out spaces
    string_fields = ["name", "model_id", "location"]

    for field in string_fields:
        field_value = getattr(request, field).strip()
        if not field_value:
            raise ValueError(f"Field '{field}' cannot be an empty string or only spaces.")

    # Check if the location exists as a directory
    if not os.path.isdir(request.location):
        raise ValueError(f"Location '{request.location}' must be a valid directory.")

    # Database validation for IDs
    with dao.get_session() as session:
        # Check if the referenced model_id exists in the database
        if not session.query(Model).filter_by(id=request.model_id.strip()).first():
            raise ValueError(f"Model with ID '{request.model_id}' does not exist.")

        # Check if an adapter with the same name already exists in the database
        if session.query(Adapter).filter_by(name=request.name.strip()).first():
            raise ValueError(f"An adapter with the name '{request.name}' already exists.")

        # Additional optional checks for fine_tuning_job_id and prompt_id
        if request.fine_tuning_job_id and not session.query(
                FineTuningJob).filter_by(id=request.fine_tuning_job_id.strip()).first():
            raise ValueError(f"Fine Tuning Job with ID '{request.fine_tuning_job_id}' does not exist.")

        if request.prompt_id and not session.query(Prompt).filter_by(id=request.prompt_id.strip()).first():
            raise ValueError(f"Prompt with ID '{request.prompt_id}' does not exist.")


def add_adapter(request: AddAdapterRequest, cml: CMLServiceApi = None,
                dao: FineTuningStudioDao = None) -> AddAdapterResponse:

    # Validate the AddAdapterRequest
    _validate_add_adapter_request(request, dao)

    response = AddAdapterResponse()

    with dao.get_session() as session:
        adapter: Adapter = Adapter(
            id=str(uuid4()),
            type=request.type,
            name=request.name,
            model_id=request.model_id,
            location=request.location,
            huggingface_name=request.huggingface_name,
            fine_tuning_job_id=request.fine_tuning_job_id,
            prompt_id=request.prompt_id,
        )
        session.add(adapter)

        response = AddAdapterResponse(
            adapter=adapter.to_protobuf(AdapterMetadata)
        )

    return response


def remove_adapter(request: RemoveAdapterRequest, cml: CMLServiceApi = None,
                   dao: FineTuningStudioDao = None) -> RemoveAdapterResponse:
    with dao.get_session() as session:
        session.execute(delete(Adapter).where(Adapter.id == request.id))
    return RemoveAdapterResponse()


def get_dataset_split_by_adapter(request: GetDatasetSplitByAdapterRequest, cml: CMLServiceApi = None,
                                 dao: FineTuningStudioDao = None) -> GetDatasetSplitByAdapterResponse:

    with dao.get_session() as session:
        try:
            row: FineTuningJob = session.query(
                FineTuningJob).join(
                Adapter, FineTuningJob.adapter_name == Adapter.name).filter(
                Adapter.id == request.adapter_id).first()
            if row is None:
                return GetDatasetSplitByAdapterResponse(
                    response=GetDatasetSplitByAdapterMetadata(
                        dataset_fraction=TRAINING_DEFAULT_DATASET_FRACTION,
                        train_test_split=TRAINING_DEFAULT_TRAIN_TEST_SPLIT))
            return GetDatasetSplitByAdapterResponse(
                response=row.to_protobuf(GetDatasetSplitByAdapterMetadata))
        except NoResultFound:
            return GetDatasetSplitByAdapterResponse(
                response=GetDatasetSplitByAdapterMetadata(
                    dataset_fraction=TRAINING_DEFAULT_DATASET_FRACTION,
                    train_test_split=TRAINING_DEFAULT_TRAIN_TEST_SPLIT))
