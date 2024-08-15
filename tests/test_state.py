import os
import tempfile
from unittest.mock import patch
from google.protobuf.json_format import ParseDict
import json

from ft.api import *

import pytest

from ft.state import (
    get_state_location,
    get_state,
    write_state,
    replace_state_field,
    DEFAULT_STATE_LOCATION
)
from ft.api import AppState, ModelType


def test_get_state_location_no_env():
    if os.environ.get("FINE_TUNING_APP_STATE_LOCATION"):
        del os.environ["FINE_TUNING_APP_STATE_LOCATION"]
    state_location = get_state_location()
    assert state_location == DEFAULT_STATE_LOCATION


def test_get_state_location_env():
    os.environ["FINE_TUNING_APP_STATE_LOCATION"] = "new/state/location.json"
    state_location = get_state_location()
    assert state_location == "new/state/location.json"


def test_get_state_empty():
    os.environ["FINE_TUNING_APP_STATE_LOCATION"] = "tests/resources/test_state_empty.json"
    state: AppState = get_state()
    assert len(state.models) == 0
    assert len(state.adapters) == 0
    assert len(state.prompts) == 0


def test_get_state_models():
    os.environ["FINE_TUNING_APP_STATE_LOCATION"] = "tests/resources/test_state_models.json"
    state: AppState = get_state()
    assert len(state.adapters) == 0
    assert len(state.models) == 1
    assert state.models[0].location == ""
    assert state.models[0].type == ModelType.MODEL_TYPE_HUGGINGFACE


@patch("ft.state.get_state_location")
def test_write_state(get_state_location):
    tmp = tempfile.NamedTemporaryFile()
    get_state_location.return_value = tmp.name
    state: AppState = ParseDict({
        "models": [
            {
                "id": "1fcb8f73-432b-4af7-abf0-1fae3e6579d5",
                "type": 0,
                "name": "Qwen/Qwen2-0.5B",
                "huggingface_model_name": "Qwen/Qwen2-0.5B"
            }
        ]
    }, AppState())
    write_state(state)

    # Open the file
    json_in = json.load(open(tmp.name))
    assert json_in.get("models") is not None
    assert json_in.get("adapters") is None
    assert json_in["models"][0]["name"] == "Qwen/Qwen2-0.5B"


@patch("ft.state.write_state")
def test_replace_state_field(write_state):
    state: AppState = AppState(
        datasets=[
            DatasetMetadata(
                id="d1"
            )
        ]
    )

    state_out = replace_state_field(state, prompts=[
        PromptMetadata(
            id="p1"
        )
    ])

    expected_state_out = AppState(
        datasets=[
            DatasetMetadata(
                id="d1"
            )
        ],
        prompts=[
            PromptMetadata(
                id="p1"
            )
        ]
    )

    assert state_out == expected_state_out
    write_state.assert_called_with(expected_state_out)


@patch("ft.state.write_state")
def test_replace_state_field_non_repeated(write_state):
    """
    Right now, we are only writing repeated composite
    messages to the top level of app state, but that might change
    in the future. This method should be resilient to simple top-level
    messages (for example, some future type of global app config).

    Because we currently don't have these high-level non-repeated messages
    in the app state, we should mark it with a test!.
    """

    some_message: DatasetMetadata = DatasetMetadata(
        id="d1",
        huggingface_name="huggingface/dataset"
    )

    with pytest.raises(KeyError):
        message_out = replace_state_field(some_message, id="d2")
