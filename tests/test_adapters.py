from unittest.mock import patch
import pytest

from ft.adapters import (
    list_adapters,
    get_adapter,
    add_adapter,
    remove_adapter,
)
from ft.api import *


def test_list_adapters():
    state: AppState = AppState(
        adapters=[
            AdapterMetadata(
                id="a1"
            )
        ]
    )
    res = list_adapters(state, ListAdaptersRequest())
    assert res.adapters[0].id == "a1"


def test_get_adapter_happy():
    state: AppState = AppState(
        adapters=[
            AdapterMetadata(
                id="ad1"
            )
        ]
    )
    req = GetAdapterRequest(id="ad1")
    res = get_adapter(state, req)
    assert res.adapter.id == "ad1"


def test_get_adapter_missing():
    state: AppState = AppState()
    with pytest.raises(AssertionError):
        res = get_adapter(state, GetAdapterRequest())


@patch("ft.adapters.write_state")
def test_add_adapter_happy(write_state):
    state: AppState = AppState()
    req = AddAdapterRequest(
        adapter=AdapterMetadata(
            id="ad1"
        )
    )
    res = add_adapter(state, req)
    write_state.assert_called_with(AppState(
        adapters=[
            AdapterMetadata(
                id="ad1"
            )
        ]
    ))


@patch("ft.adapters.replace_state_field")
def test_remove_adapter_happy(replace_state_field):
    state: AppState = AppState(
        adapters=[
            AdapterMetadata(
                id="ad1"
            ),
            AdapterMetadata(
                id="ad2"
            )
        ]
    )
    req = RemoveAdapterRequest(id="ad1")
    res = remove_adapter(state, req)
    replace_state_field.assert_called_with(
        state,
        adapters=[
            AdapterMetadata(
                id="ad2"
            )
        ]
    )
