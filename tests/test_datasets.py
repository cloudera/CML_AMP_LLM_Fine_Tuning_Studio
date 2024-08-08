from unittest.mock import patch
import pytest

from ft.datasets import (
    list_datasets,
    get_dataset,
    remove_dataset,
)
from ft.api import *


def test_list_datasets():
    state: AppState = AppState(
        datasets=[
            DatasetMetadata(
                id="d1"
            )
        ]
    )
    res = list_datasets(state, ListDatasetsRequest())
    assert res.datasets[0].id == "d1"


def test_get_dataset_happy():
    state: AppState = AppState(
        datasets=[
            DatasetMetadata(
                id="d1"
            )
        ]
    )
    req = GetDatasetRequest(id="d1")
    res = get_dataset(state, req)
    assert res.dataset.id == "d1"


def test_get_dataset_missing():
    state: AppState = AppState()
    with pytest.raises(AssertionError):
        res = get_dataset(state, GetDatasetRequest())


@patch("ft.datasets.write_state")
def test_remove_dataset_happy(write_state):
    state: AppState = AppState(
        datasets=[
            DatasetMetadata(
                id="d1"
            ),
            DatasetMetadata(
                id="d2"
            )
        ]
    )
    req = RemoveDatasetRequest(id="d1")
    res = remove_dataset(state, req)
    write_state.assert_called_with(AppState(
        datasets=[
            DatasetMetadata(
                id="d2"
            )
        ]
    ))


@patch("ft.datasets.write_state")
def test_remove_dataset_remove_prompts(write_state):
    state: AppState = AppState(
        datasets=[
            DatasetMetadata(
                id="d1"
            ),
            DatasetMetadata(
                id="d2"
            )
        ],
        prompts=[
            PromptMetadata(
                id="p1",
                dataset_id="d1"
            ),
            PromptMetadata(
                id="p2",
                dataset_id="d2"
            )
        ]
    )
    req = RemoveDatasetRequest(id="d1", remove_prompts=True)
    res = remove_dataset(state, req)
    write_state.assert_called_with(AppState(
        datasets=[
            DatasetMetadata(
                id="d2"
            )
        ],
        prompts=[
            PromptMetadata(
                id="p2",
                dataset_id="d2"
            )
        ]
    ))
