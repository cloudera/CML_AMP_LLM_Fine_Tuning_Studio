import pytest
from sqlalchemy.exc import NoResultFound
import json

from ft.datasets import (
    list_datasets,
    get_dataset,
    add_dataset,
    remove_dataset,
    load_dataset_into_memory,
    extract_features_from_csv
)
from ft.api import *
from ft.db.dao import FineTuningStudioDao
from ft.db.model import Dataset, Prompt
from unittest.mock import patch
import datasets

# Test: Listing datasets


def test_list_datasets():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Dataset(id="d1"))
        session.add(Dataset(id="d2", features=json.dumps(["f1", "f2"])))
        session.commit()

    res = list_datasets(ListDatasetsRequest(), dao=test_dao)
    assert res.datasets[0].id == "d1"
    assert res.datasets[1].id == "d2"
    assert res.datasets[1].features == '["f1", "f2"]'

# Test: Getting a dataset by ID (happy path)


def test_get_dataset_happy():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Dataset(id="d1"))
        session.add(Dataset(id="d2", type=DatasetType.HUGGINGFACE))
        session.commit()

    req = GetDatasetRequest(id="d2")
    res = get_dataset(req, dao=test_dao)
    assert res.dataset.id == "d2"
    assert res.dataset.type == "huggingface"

# Test: Getting a dataset by ID (dataset missing)


def test_get_dataset_missing():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with pytest.raises(NoResultFound):
        get_dataset(GetDatasetRequest(id="d1"), dao=test_dao)

# Test: Removing a dataset (happy path)


def test_remove_dataset_happy():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Dataset(id="d1"))
        session.add(Dataset(id="d2", type=DatasetType.HUGGINGFACE))
        session.add(Prompt(id="p1", dataset_id="d1"))
        session.add(Prompt(id="p2", dataset_id="d2"))
        session.commit()

    req = RemoveDatasetRequest(id="d1")
    res = remove_dataset(req, dao=test_dao)

    with test_dao.get_session() as session:
        assert len(list(session.query(Dataset).all())) == 1
        assert len(list(session.query(Prompt).all())) == 2

# Test: Removing a dataset with prompts


def test_remove_dataset_remove_prompts():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Dataset(id="d1"))
        session.add(Dataset(id="d2", type=DatasetType.HUGGINGFACE))
        session.add(Prompt(id="p1", dataset_id="d1"))
        session.add(Prompt(id="p2", dataset_id="d2"))
        session.commit()

    req = RemoveDatasetRequest(id="d1", remove_prompts=True)
    res = remove_dataset(req, dao=test_dao)

    with test_dao.get_session() as session:
        assert len(list(session.query(Dataset).all())) == 1
        assert len(list(session.query(Prompt).all())) == 1

# Test: Adding a dataset (happy path)


@patch("ft.datasets.load_dataset_builder")
@patch("ft.datasets.uuid4")
def test_add_dataset_happy(uuid4_mock, load_dataset_builder_mock):
    uuid4_mock.return_value = "d3"

    # Mock the dataset info returned by load_dataset_builder
    mock_dataset_builder = load_dataset_builder_mock.return_value
    mock_dataset_builder.info.features = {"f1": "value1", "f2": "value2"}
    mock_dataset_builder.info.description = "Test dataset description"

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Dataset(id="d1", type=DatasetType.PROJECT_CSV, location="dataset.csv"))
        session.add(Dataset(id="d2", type=DatasetType.PROJECT, location="path/to/dataset"))
        session.commit()

    request = AddDatasetRequest(
        type=DatasetType.HUGGINGFACE,
        huggingface_name="test_dataset"
    )

    res = add_dataset(request, dao=test_dao)

    assert res.dataset.id == "d3"
    assert res.dataset.name == "test_dataset"
    assert res.dataset.features == '["f1", "f2"]'
    assert res.dataset.description == "Test dataset description"

# Test: Adding a dataset with a missing required field


def test_add_dataset_missing_required_field():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    request = AddDatasetRequest(
        type=DatasetType.HUGGINGFACE,
        huggingface_name=""
    )

    with pytest.raises(ValueError) as excinfo:
        add_dataset(request, dao=test_dao)
    assert "Field 'huggingface_name' is required in AddDatasetRequest." in str(excinfo.value)

# Test: Adding a dataset that already exists


@patch("ft.datasets.load_dataset_builder")
def test_add_dataset_already_exists(load_dataset_builder_mock):
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Dataset(id="d1", type=DatasetType.HUGGINGFACE, huggingface_name="existing_dataset"))
        session.commit()

    request = AddDatasetRequest(
        type=DatasetType.HUGGINGFACE,
        huggingface_name="existing_dataset"
    )

    with pytest.raises(ValueError) as excinfo:
        add_dataset(request, dao=test_dao)
    assert "Dataset with name 'existing_dataset' of type huggingface already exists." in str(excinfo.value)


@patch("ft.datasets.open")
@patch("ft.datasets.csv")
@patch("ft.datasets.next")
@patch("ft.datasets.uuid4")
def test_add_dataset_local_csv_happy(uuid4, next, csv, open):
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    uuid4.return_value = "ds1"
    next.return_value = ["feature1", "feature2", "feature3"]

    request = AddDatasetRequest(
        type=DatasetType.PROJECT_CSV,
        name="my dataset",
        location="path/to/dataset.csv"
    )

    res: DatasetMetadata = add_dataset(request, dao=test_dao).dataset
    assert res.id == "ds1"
    assert res.type == DatasetType.PROJECT_CSV
    assert res.features == '["feature1", "feature2", "feature3"]'


@patch("ft.datasets.open")
@patch("ft.datasets.csv")
@patch("ft.datasets.next")
@patch("ft.datasets.uuid4")
def test_add_dataset_local_csv_bad_path(uuid4, next, csv, open):
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    uuid4.return_value = "ds1"
    next.return_value = ["feature1", "feature2", "feature3"]

    request = AddDatasetRequest(
        type=DatasetType.PROJECT_CSV,
        name="my dataset",
        location="path/to/dataset.json"
    )

    with pytest.raises(ValueError) as excinfo:
        add_dataset(request, dao=test_dao)
    assert "Dataset path/to/dataset.json does not end with .csv" in str(excinfo.value)


def test_add_dataset_local_csv_already_exists():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Dataset(id="d1", type=DatasetType.PROJECT_CSV, name="my dataset", location="path/to/dataset.csv"))
        session.commit()

    request = AddDatasetRequest(
        type=DatasetType.PROJECT_CSV,
        name="my dataset",
        location="path/to/dataset.csv"
    )

    with pytest.raises(ValueError) as excinfo:
        add_dataset(request, dao=test_dao)
    assert "Dataset with name 'my dataset' of type project_csv already exists." in str(excinfo.value)


def test_load_dataset_into_memory_csv_dataset_happy():
    csv_path = "tests/resources/test_dataset_csv_healthy.csv"

    dataset: DatasetMetadata = DatasetMetadata(
        type=DatasetType.PROJECT_CSV,
        location=csv_path
    )

    dataset_out: datasets.DatasetDict = load_dataset_into_memory(dataset)
    assert isinstance(dataset_out, datasets.DatasetDict)
    assert len(dataset_out["train"]) == 2


def test_extract_features_from_csv_happy():
    csv_path = "tests/resources/test_dataset_csv_healthy.csv"

    features = extract_features_from_csv(csv_path)
    assert features == ["input", "output", "score"]


@patch("ft.datasets.uuid4")
def test_add_dataset_csv_e2e_happy(uuid4):
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    uuid4.return_value = "ds1"

    request: AddDatasetRequest = AddDatasetRequest(
        type=DatasetType.PROJECT_CSV,
        location="tests/resources/test_dataset_csv_healthy.csv",
        name="Custom CSV Dataset"
    )

    dataset: DatasetMetadata = add_dataset(request, dao=test_dao).dataset

    assert dataset.id == "ds1"
    assert dataset.features == json.dumps(["input", "output", "score"])
    assert dataset.location == "tests/resources/test_dataset_csv_healthy.csv"
    assert dataset.name == "Custom CSV Dataset"
