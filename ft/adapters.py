

from ft.api import *

from cmlapi import CMLServiceApi

from uuid import uuid4

from typing import List

from ft.db.dao import FineTuningStudioDao
from ft.db.model import Adapter

from sqlalchemy import delete


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


def add_adapter(request: AddAdapterRequest, cml: CMLServiceApi = None,
                dao: FineTuningStudioDao = None) -> AddAdapterResponse:

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
