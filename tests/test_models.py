import pytest
import unittest
from unittest.mock import patch, MagicMock
from sqlalchemy.exc import NoResultFound

from ft.models import (
    list_models,
    get_model,
    add_model,
    remove_model,
    export_model
)
from ft.api import *

from ft.db.dao import FineTuningStudioDao
from ft.db.model import Model


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
        get_model(GetModelRequest(id="a3"), dao=test_dao)


def test_remove_model_happy():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Model(id="m1"))
        session.add(Model(id="m2"))

    remove_model(RemoveModelRequest(id="m1"), dao=test_dao)
    with test_dao.get_session() as session:
        assert len(session.query(Model).all()) == 1


@patch("ft.models.HfApi.model_info")
def test_add_model_huggingface_happy(mock_model_info):
    mock_model_info.return_value = MagicMock()  # Mock the ModelInfo response

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    request = AddModelRequest(
        type=ModelType.HUGGINGFACE,
        huggingface_name="mock_model"
    )

    res = add_model(request, dao=test_dao)
    assert res.model.name == "mock_model"

    with test_dao.get_session() as session:
        models = session.query(Model).all()
        assert len(models) == 1
        assert models[0].huggingface_model_name == "mock_model"


def test_add_model_huggingface_missing_name():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    request = AddModelRequest(
        type=ModelType.HUGGINGFACE,
        huggingface_name=""
    )

    with pytest.raises(ValueError, match="Hugging Face model name cannot be an empty string or only spaces."):
        add_model(request, dao=test_dao)


@patch("ft.models.HfApi.model_info")
def test_add_model_huggingface_duplicate_name(mock_model_info):
    mock_model_info.return_value = MagicMock()

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Model(id="m1", huggingface_model_name="mock_model"))

    request = AddModelRequest(
        type=ModelType.HUGGINGFACE,
        huggingface_name="mock_model"
    )

    with pytest.raises(ValueError, match="Model with name 'mock_model' already exists."):
        add_model(request, dao=test_dao)


def test_add_model_invalid_type():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    request = AddModelRequest(
        type="INVALID_TYPE",
        huggingface_name="mock_model"
    )

    with pytest.raises(ValueError, match="ERROR: Cannot import model of this type."):
        add_model(request, dao=test_dao)


class TestModelExports(unittest.TestCase):

    @patch("ft.models.export_model_registry_model")
    def test_export_model_type_registry(self, export_to_registry):
        request: ExportModelRequest = ExportModelRequest(
            type=ModelExportType.MODEL_REGISTRY,
            base_model_id="id1"
        )
        response = export_model(request, None, None)
        export_to_registry.assert_called_once()

    @patch("ft.models.deploy_cml_model")
    def test_export_model_type_cml(self, export_to_cml):
        request: ExportModelRequest = ExportModelRequest(
            type=ModelExportType.CML_MODEL,
            base_model_id="id1"
        )
        response = export_model(request, None, None)
        export_to_cml.assert_called_once()

    def test_export_model_type_not_supported(self):
        request: ExportModelRequest = ExportModelRequest(
            type="bad_type",
            base_model_id="id1"
        )

        with self.assertRaises(ValueError) as context:
            response = export_model(request, None, None)

        assert str(context.exception) == "Model export of type 'bad_type' is not supported."
