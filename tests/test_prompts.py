from unittest.mock import patch
import pytest

from ft.prompts import (
    list_prompts,
    get_prompt,
    add_prompt,
    remove_prompt,
)
from ft.api import *


def test_list_prompts():
    state: AppState = AppState(
        prompts=[
            PromptMetadata(
                id="p1"
            )
        ]
    )
    res = list_prompts(state, ListPromptsRequest())
    assert res.prompts[0].id == "p1"


def test_get_prompt_happy():
    state: AppState = AppState(
        prompts=[
            PromptMetadata(
                id="p1"
            )
        ]
    )
    req = GetPromptRequest(id="p1")
    res = get_prompt(state, req)
    assert res.prompt.id == "p1"


def test_get_prompt_missing():
    state: AppState = AppState()
    with pytest.raises(AssertionError):
        res = get_prompt(state, GetPromptRequest())


@patch("ft.prompts.write_state")
def test_add_prompt_happy(write_state):
    state: AppState = AppState()
    req = AddPromptRequest(
        prompt=PromptMetadata(
            id="p1"
        )
    )
    res = add_prompt(state, req)
    write_state.assert_called_with(AppState(
        prompts=[
            PromptMetadata(
                id="p1"
            )
        ]
    ))


@patch("ft.prompts.write_state")
def test_remove_prompt_happy(write_state):
    state: AppState = AppState(
        prompts=[
            PromptMetadata(
                id="p1"
            ),
            PromptMetadata(
                id="p2"
            )
        ]
    )
    req = RemovePromptRequest(id="p1")
    res = remove_prompt(state, req)
    write_state.assert_called_with(AppState(
        prompts=[
            PromptMetadata(
                id="p2"
            )
        ]
    ))
