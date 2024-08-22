import pytest
from sqlalchemy.exc import NoResultFound
import json

from ft.datasets import (
    list_datasets,
    get_dataset,
    remove_dataset,
)
from ft.api import *

from ft.db.dao import FineTuningStudioDao
from ft.db.model import Dataset, Prompt


def test_list_datasets():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Dataset(id="d1"))
        session.add(Dataset(id="d2", features=json.dumps(["f1", "f2"])))

    res = list_datasets(ListDatasetsRequest(), dao=test_dao)
    assert res.datasets[0].id == "d1"
    assert res.datasets[1].id == "d2"
    assert res.datasets[1].features == '["f1", "f2"]'


def test_get_dataset_happy():

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Dataset(id="d1"))
        session.add(Dataset(id="d2", type=DatasetType.HUGGINGFACE))

    req = GetDatasetRequest(id="d2")
    res = get_dataset(req, dao=test_dao)
    assert res.dataset.id == "d2"
    assert res.dataset.type == "huggingface"


def test_get_dataset_missing():

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with pytest.raises(NoResultFound):
        res = get_dataset(GetDatasetRequest(id="d1"), dao=test_dao)


def test_remove_dataset_happy():

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Dataset(id="d1"))
        session.add(Dataset(id="d2", type=DatasetType.HUGGINGFACE))
        session.add(Prompt(id="p1", dataset_id="d1"))
        session.add(Prompt(id="p2", dataset_id="d2"))

    req = RemoveDatasetRequest(id="d1")
    res = remove_dataset(req, dao=test_dao)

    with test_dao.get_session() as session:
        assert len(list(session.query(Dataset).all())) == 1
        assert len(list(session.query(Prompt).all())) == 2


def test_remove_dataset_remove_prompts():

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Dataset(id="d1"))
        session.add(Dataset(id="d2", type=DatasetType.HUGGINGFACE))
        session.add(Prompt(id="p1", dataset_id="d1"))
        session.add(Prompt(id="p2", dataset_id="d2"))

    req = RemoveDatasetRequest(id="d1", remove_prompts=True)
    res = remove_dataset(req, dao=test_dao)

    with test_dao.get_session() as session:
        assert len(list(session.query(Dataset).all())) == 1
        assert len(list(session.query(Prompt).all())) == 1
