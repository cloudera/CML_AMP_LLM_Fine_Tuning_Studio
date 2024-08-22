import pytest

from ft.prompts import (
    list_prompts,
    get_prompt,
    add_prompt,
    remove_prompt,
)
from ft.api import *

from ft.db.dao import FineTuningStudioDao
from ft.db.model import Prompt

from sqlalchemy.exc import NoResultFound


def test_list_prompts():

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Prompt(id="p1"))
        session.add(Prompt(id="p2"))

    res = list_prompts(ListPromptsRequest(), dao=test_dao)
    assert res.prompts[0].id == "p1"
    assert len(res.prompts) == 2


def test_get_prompt_happy():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Prompt(id="p1"))
        session.add(Prompt(id="p2"))

    res = get_prompt(GetPromptRequest(id="p2"), dao=test_dao)
    assert res.prompt.id == "p2"


def test_get_prompt_missing():

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Prompt(id="p1"))
        session.add(Prompt(id="p2"))

    with pytest.raises(NoResultFound):
        res = get_prompt(GetPromptRequest(id="p3"), dao=test_dao)


def test_add_prompt_happy():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Prompt(id="p1"))
        session.add(Prompt(id="p2"))

    res = add_prompt(AddPromptRequest(
        prompt=PromptMetadata(
            id="p3"
        )
    ), dao=test_dao)
    assert res.prompt.id == "p3"

    with test_dao.get_session() as session:
        assert len(session.query(Prompt).all()) == 3


def test_remove_prompt_happy():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Prompt(id="p1"))
        session.add(Prompt(id="p2"))

    res = remove_prompt(RemovePromptRequest(id="p1"), dao=test_dao)

    with test_dao.get_session() as session:
        assert len(session.query(Prompt).all()) == 1
        assert session.query(Prompt).one().id == "p2"
