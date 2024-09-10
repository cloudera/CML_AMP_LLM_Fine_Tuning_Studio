from unittest.mock import patch
import pytest
from ft.adapters import (
    list_adapters,
    get_adapter,
    add_adapter,
    remove_adapter,
    _validate_add_adapter_request,
    get_dataset_split_by_adapter
)
from ft.api import *

from ft.db.dao import FineTuningStudioDao
from ft.db.model import Adapter, Model, FineTuningJob
from ft.consts import TRAINING_DEFAULT_DATASET_FRACTION, TRAINING_DEFAULT_TRAIN_TEST_SPLIT
from sqlalchemy.exc import NoResultFound


def test_list_adapters():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Adapter(id="a1"))
        session.add(Adapter(id="a2"))
        session.commit()

    res = list_adapters(ListAdaptersRequest(), dao=test_dao)
    assert len(res.adapters) == 2


def test_get_adapter_happy():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Adapter(id="a1"))
        session.add(Adapter(id="a2"))
        session.commit()

    req = GetAdapterRequest(id="a1")
    res = get_adapter(req, dao=test_dao)
    assert res.adapter.id == "a1"


def test_get_adapter_missing():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Adapter(id="a1"))
        session.add(Adapter(id="a2"))
        session.commit()

    with pytest.raises(NoResultFound):
        get_adapter(GetAdapterRequest(id="a3"), dao=test_dao)

# Test: Successful adapter addition


@patch("ft.adapters.uuid4")
@patch("ft.adapters.os.path.isdir")
def test_add_adapter_happy_case(isdir_mock, uuid4_mock):
    isdir_mock.return_value = True
    uuid4_mock.return_value = "a3"

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    # Insert necessary models into the database
    with test_dao.get_session() as session:
        session.add(Model(id="model-id"))
        session.commit()

    request = AddAdapterRequest(
        name="my adapter", model_id="model-id", location="/data/adapter"
    )

    response = add_adapter(request, dao=test_dao)

    assert response.adapter.id == "a3"
    assert response.adapter.name == "my adapter"

# Test: Missing required field


def test_add_adapter_missing_required_field():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    request = AddAdapterRequest(
        name="", model_id="model-id", location="/data/adapter"
    )

    with pytest.raises(ValueError) as excinfo:
        _validate_add_adapter_request(request, test_dao)
    assert "Field 'name' is required in AddAdapterRequest." in str(excinfo.value)

# Test: Empty string fields after stripping spaces


def test_add_adapter_empty_string_field():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    request = AddAdapterRequest(
        name="   ", model_id="model-id", location="/data/adapter"
    )

    with pytest.raises(ValueError) as excinfo:
        _validate_add_adapter_request(request, test_dao)
    assert "Field 'name' cannot be an empty string or only spaces." in str(excinfo.value)

# Test: Invalid directory location


@patch("ft.adapters.os.path.isdir")
def test_add_adapter_invalid_directory(isdir_mock):
    isdir_mock.return_value = False

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    request = AddAdapterRequest(
        name="my adapter", model_id="model-id", location="/invalid/dir"
    )

    with pytest.raises(ValueError) as excinfo:
        _validate_add_adapter_request(request, test_dao)
    assert "Location '/invalid/dir' must be a valid directory." in str(excinfo.value)

# Test: Model ID does not exist in the database


@patch("ft.adapters.os.path.isdir")
def test_add_adapter_model_id_not_exist(isdir_mock):
    isdir_mock.return_value = True

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    request = AddAdapterRequest(
        name="my adapter", model_id="non-existent-model-id", location="/data/adapter"
    )

    with pytest.raises(ValueError) as excinfo:
        _validate_add_adapter_request(request, test_dao)
    assert "Model with ID 'non-existent-model-id' does not exist." in str(excinfo.value)

# Test: Adapter with the same name already exists


@patch("ft.adapters.os.path.isdir")
@patch("ft.adapters.uuid4", return_value="a1")  # Ensure the ID is set properly
def test_add_adapter_name_exists(isdir_mock, uuid4_mock):
    isdir_mock.return_value = True

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Model(id="model-id"))
        session.add(Adapter(id="a1", name="my adapter", model_id="model-id", location="/data/adapter"))
        session.commit()

    request = AddAdapterRequest(
        name="my adapter", model_id="model-id", location="/data/adapter"
    )

    with pytest.raises(ValueError) as excinfo:
        _validate_add_adapter_request(request, test_dao)
    assert "An adapter with the name 'my adapter' already exists." in str(excinfo.value)

# Test: Fine Tuning Job ID does not exist in the database


@patch("ft.adapters.os.path.isdir")
def test_add_adapter_fine_tuning_job_id_not_exist(isdir_mock):
    isdir_mock.return_value = True

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Model(id="model-id"))
        session.commit()

    request = AddAdapterRequest(
        name="my adapter",
        model_id="model-id",
        location="/data/adapter",
        fine_tuning_job_id="non-existent-fine-tuning-job-id",
    )

    with pytest.raises(ValueError) as excinfo:
        _validate_add_adapter_request(request, test_dao)
    assert "Fine Tuning Job with ID 'non-existent-fine-tuning-job-id' does not exist." in str(excinfo.value)

# Test: Prompt ID does not exist in the database


@patch("ft.adapters.os.path.isdir")
def test_add_adapter_prompt_id_not_exist(isdir_mock):
    isdir_mock.return_value = True

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Model(id="model-id"))
        session.commit()

    request = AddAdapterRequest(
        name="my adapter",
        model_id="model-id",
        location="/data/adapter",
        prompt_id="non-existent-prompt-id",
    )

    with pytest.raises(ValueError) as excinfo:
        _validate_add_adapter_request(request, test_dao)
    assert "Prompt with ID 'non-existent-prompt-id' does not exist." in str(excinfo.value)


def test_remove_adapter_happy():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Adapter(id="a1"))
        session.add(Adapter(id="a2"))
        session.commit()

    res = remove_adapter(RemoveAdapterRequest(id="a1"), dao=test_dao)
    with test_dao.get_session() as session:
        assert len(session.query(Adapter).all()) == 1


def test_get_dataset_split_by_adapter_with_row():
    # Setup mock
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(FineTuningJob(id="abcd", train_test_split=0.8, dataset_fraction=0.7, adapter_name="adapter"))
        session.add(Adapter(id="ada", name="adapter"))
        session.commit()

    req = GetDatasetSplitByAdapterRequest(adapter_id="ada")
    res = get_dataset_split_by_adapter(req, dao=test_dao)
    assert int(100 * res.response.dataset_fraction) == int(100 * 0.7)
    assert int(100 * res.response.train_test_split) == int(100 * 0.8)


# Test when no rows are returned
def test_get_dataset_split_by_adapter_no_rows():
    # Setup mock
    # Setup mock
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(FineTuningJob(id="abcd", train_test_split=0.1, dataset_fraction=0.2, adapter_name="adapter"))
        session.add(Adapter(id="ada2", name="adapter2"))
        session.commit()

    req = GetDatasetSplitByAdapterRequest(adapter_id="ada2")
    res = get_dataset_split_by_adapter(req, dao=test_dao)
    assert res.response.dataset_fraction != 0.1
    assert res.response.train_test_split != 0.2
    assert res.response.dataset_fraction == TRAINING_DEFAULT_DATASET_FRACTION
    assert res.response.train_test_split == TRAINING_DEFAULT_TRAIN_TEST_SPLIT
