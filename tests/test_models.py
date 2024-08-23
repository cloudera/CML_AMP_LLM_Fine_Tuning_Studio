import pytest

from ft.models import (
    list_models,
    get_model,
    remove_model,
)
from ft.api import *

from ft.db.dao import FineTuningStudioDao
from ft.db.model import Model

from sqlalchemy.exc import NoResultFound


def test_list_models():

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Model(id="m1"))
        session.add(Model(id="m2"))

    res = list_models(ListModelsRequest(), dao=test_dao)
    assert len(res.models) == 2


def test_get_model_happy():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Model(id="m1"))
        session.add(Model(id="m2"))

    req = GetModelRequest(id="m1")
    res = get_model(req, dao=test_dao)
    assert res.model.id == "m1"


def test_get_model_missing():

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Model(id="m1"))
        session.add(Model(id="m2"))

    with pytest.raises(NoResultFound):
        res = get_model(GetModelRequest(id="a3"), dao=test_dao)


def test_remove_model_happy():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Model(id="m1"))
        session.add(Model(id="m2"))

    res = remove_model(RemoveModelRequest(id="m1"), dao=test_dao)
    with test_dao.get_session() as session:
        assert len(session.query(Model).all()) == 1
