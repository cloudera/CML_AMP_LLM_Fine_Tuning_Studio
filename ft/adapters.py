

from ft.api import *

from cmlapi import CMLServiceApi


from ft.state import write_state, replace_state_field

from uuid import uuid4


def list_adapters(state: AppState, request: ListAdaptersRequest, cml: CMLServiceApi = None) -> ListAdaptersResponse:
    """
    Right now we don't do any filtering in this op, but we might in the future.
    """
    return ListAdaptersResponse(
        adapters=state.adapters
    )


def get_adapter(state: AppState, request: GetAdapterRequest, cml: CMLServiceApi = None) -> GetAdapterResponse:
    adapters = list(filter(lambda x: x.id == request.id, state.adapters))
    assert len(adapters) == 1
    return GetAdapterResponse(
        adapter=adapters[0]
    )


def add_adapter(state: AppState, request: AddAdapterRequest, cml: CMLServiceApi = None) -> AddAdapterResponse:

    # TODO: see if there is a cleaner way to merge protobuf messages together
    adapter_md: AdapterMetadata = AdapterMetadata(
        id=str(uuid4()),
        type=request.type,
        name=request.name,
        model_id=request.model_id,
        location=request.location,
        huggingface_name=request.huggingface_name,
        fine_tuning_job_id=request.fine_tuning_job_id,
        prompt_id=request.prompt_id,
    )

    state.adapters.append(adapter_md)
    write_state(state)
    return AddAdapterResponse(adapter=adapter_md)


def remove_adapter(state: AppState, request: RemoveAdapterRequest, cml: CMLServiceApi = None) -> RemoveAdapterResponse:
    adapters = list(filter(lambda x: not x.id == request.id, state.adapters))
    state = replace_state_field(state, adapters=adapters)
    return RemoveAdapterResponse()
