import pytest
from sqlalchemy.exc import NoResultFound

from ft.prompts import (
    list_prompts,
    get_prompt,
    add_prompt,
    remove_prompt,
)
from ft.api import *
from ft.db.dao import FineTuningStudioDao
from ft.db.model import Prompt


def test_list_prompts():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Prompt(id="p1", name="Prompt 1"))
        session.add(Prompt(id="p2", name="Prompt 2"))

    res = list_prompts(ListPromptsRequest(), dao=test_dao)
    assert len(res.prompts) == 2
    assert res.prompts[0].id == "p1"
    assert res.prompts[1].id == "p2"


def test_get_prompt_happy():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Prompt(id="p1", name="Prompt 1"))
        session.add(Prompt(id="p2", name="Prompt 2"))

    res = get_prompt(GetPromptRequest(id="p2"), dao=test_dao)
    assert res.prompt.id == "p2"
    assert res.prompt.name == "Prompt 2"


def test_get_prompt_missing():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Prompt(id="p1", name="Prompt 1"))
        session.add(Prompt(id="p2", name="Prompt 2"))

    with pytest.raises(NoResultFound):
        get_prompt(GetPromptRequest(id="p3"), dao=test_dao)


def test_add_prompt_happy():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    request = AddPromptRequest(
        prompt=PromptMetadata(
            id="p3",
            name="Prompt 3",
            dataset_id="ds1",
            prompt_template="Template",
            input_template="Input",
            completion_template="Completion"
        )
    )

    res = add_prompt(request, dao=test_dao)
    assert res.prompt.id == "p3"
    assert res.prompt.name == "Prompt 3"

    with test_dao.get_session() as session:
        prompts = session.query(Prompt).all()
        assert len(prompts) == 1
        assert prompts[0].id == "p3"


def test_add_prompt_missing_field():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    request = AddPromptRequest(
        prompt=PromptMetadata(
            id="p4",
            name="",  # Missing name (empty)
            dataset_id="ds1",
            prompt_template="Template",
            input_template="Input",
            completion_template="Completion"
        )
    )

    with pytest.raises(ValueError, match="Field 'name' is required in PromptMetadata."):
        add_prompt(request, dao=test_dao)


def test_add_prompt_duplicate_name():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Prompt(id="p1", name="Duplicate Prompt"))

    request = AddPromptRequest(
        prompt=PromptMetadata(
            id="p2",
            name="Duplicate Prompt",  # Duplicate name
            dataset_id="ds1",
            prompt_template="Template",
            input_template="Input",
            completion_template="Completion"
        )
    )

    with pytest.raises(ValueError, match="Prompt name 'Duplicate Prompt' already exists."):
        add_prompt(request, dao=test_dao)


def test_remove_prompt_happy():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Prompt(id="p1", name="Prompt 1"))
        session.add(Prompt(id="p2", name="Prompt 2"))

    remove_prompt(RemovePromptRequest(id="p1"), dao=test_dao)

    with test_dao.get_session() as session:
        prompts = session.query(Prompt).all()
        assert len(prompts) == 1
        assert prompts[0].id == "p2"
        assert prompts[0].name == "Prompt 2"
