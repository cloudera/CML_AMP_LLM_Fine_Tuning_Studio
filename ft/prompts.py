

from ft.api import *

from cmlapi import CMLServiceApi

from ft.state import write_state, replace_state_field


def list_prompts(state: AppState, request: ListPromptsRequest, cml: CMLServiceApi = None) -> ListPromptsResponse:
    """
    Right now we don't do any filtering in this op, but we might in the future.
    """
    return ListPromptsResponse(
        prompts=state.prompts
    )


def get_prompt(state: AppState, request: GetPromptRequest, cml: CMLServiceApi = None) -> GetPromptResponse:
    prompts = list(filter(lambda x: x.id == request.id, state.prompts))
    assert len(prompts) == 1
    return GetPromptResponse(
        prompt=prompts[0]
    )


def add_prompt(state: AppState, request: AddPromptRequest, cml: CMLServiceApi = None) -> AddPromptResponse:
    state.prompts.append(request.prompt)
    write_state(state)
    return AddPromptResponse()


def remove_prompt(state: AppState, request: RemovePromptRequest, cml: CMLServiceApi = None) -> RemovePromptResponse:
    prompts = list(filter(lambda x: not x.id == request.id, state.prompts))
    state = replace_state_field(state, prompts=prompts)
    return RemovePromptResponse()
