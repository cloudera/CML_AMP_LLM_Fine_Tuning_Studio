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


@patch("ft.datasets.replace_state_field")
def test_remove_dataset_happy(replace_state_field):
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
    replace_state_field.assert_any_call(state,
                                        datasets=[
                                            DatasetMetadata(
                                                id="d2"
                                            )
                                        ]
                                        )


@patch("ft.datasets.replace_state_field")
def test_remove_dataset_remove_prompts(replace_state_field):
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

    replace_state_field.return_value = state

    req = RemoveDatasetRequest(id="d1", remove_prompts=True)
    res = remove_dataset(state, req)
    replace_state_field.assert_any_call(state,
                                        datasets=[
                                            DatasetMetadata(
                                                id="d2"
                                            )
                                        ]
                                        )
    replace_state_field.assert_any_call(state,
                                        prompts=[
                                            PromptMetadata(
                                                id="p2",
                                                dataset_id="d2"
                                            )
                                        ]
                                        )
