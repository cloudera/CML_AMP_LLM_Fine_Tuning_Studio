from unittest.mock import patch
import pytest

from ft.adapters import (
    list_adapters,
    get_adapter,
    add_adapter,
    remove_adapter,
)
from ft.api import *

from ft.db.dao import FineTuningStudioDao
from ft.db.model import Adapter

from sqlalchemy.exc import NoResultFound


def test_list_adapters():

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Adapter(id="a1"))
        session.add(Adapter(id="a2"))

    res = list_adapters(ListAdaptersRequest(), dao=test_dao)
    assert len(res.adapters) == 2


def test_get_adapter_happy():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Adapter(id="a1"))
        session.add(Adapter(id="a2"))

    req = GetAdapterRequest(id="a1")
    res = get_adapter(req, dao=test_dao)
    assert res.adapter.id == "a1"


def test_get_adapter_missing():

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Adapter(id="a1"))
        session.add(Adapter(id="a2"))

    with pytest.raises(NoResultFound):
        res = get_adapter(GetAdapterRequest(id="a3"), dao=test_dao)


@patch("ft.adapters.uuid4")
def test_add_adapter_happy(uuid4):
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Adapter(id="a1"))
        session.add(Adapter(id="a2"))

    uuid4.return_value = "a3"

    res = add_adapter(AddAdapterRequest(name="my adapter"), dao=test_dao)
    assert res.adapter.id == "a3"

    with test_dao.get_session() as session:
        assert len(session.query(Adapter).where(Adapter.name == "my adapter").all()) == 1


def test_remove_adapter_happy():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Adapter(id="a1"))
        session.add(Adapter(id="a2"))

    res = remove_adapter(RemoveAdapterRequest(id="a1"), dao=test_dao)
    with test_dao.get_session() as session:
        assert len(session.query(Adapter).all()) == 1
