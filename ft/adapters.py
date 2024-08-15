

from ft.api import *

from cmlapi import CMLServiceApi


from ft.state import write_state, replace_state_field


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
    state.adapters.append(request.adapter)
    write_state(state)
    return AddAdapterResponse(adapter=request.adapter)


def remove_adapter(state: AppState, request: RemoveAdapterRequest, cml: CMLServiceApi = None) -> RemoveAdapterResponse:
    adapters = list(filter(lambda x: not x.id == request.id, state.adapters))
    state = replace_state_field(state, adapters=adapters)
    return RemoveAdapterResponse()
