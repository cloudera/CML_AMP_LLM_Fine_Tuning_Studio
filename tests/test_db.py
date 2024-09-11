from unittest.mock import patch
import unittest

from ft.db.dao import get_sqlite_db_location, FineTuningStudioDao
from ft.consts import DEFAULT_SQLITE_DB_LOCATION
from ft.db.utils import (
    export_to_dict,
    import_from_dict
)
from ft.db.model import *
from ft.api import *
from typing import List


@patch("ft.db.dao.os.environ.get")
def test_get_sqlite_db_location_no_env_var(get):
    get.return_value = None

    out = get_sqlite_db_location()
    assert out == DEFAULT_SQLITE_DB_LOCATION
    get.assert_called_once()


@patch("ft.db.dao.os.environ.get")
def test_get_sqlite_db_location_env_var(get):
    get.return_value = "test/app/state.db"

    out = get_sqlite_db_location()
    assert not out == DEFAULT_SQLITE_DB_LOCATION
    assert 2 == get.call_count
    assert out == "test/app/state.db"


@patch("ft.db.dao.create_engine")
@patch("ft.db.dao.get_sqlite_db_location")
def test_db_dao_init_no_url(get_sqlite_db_location, create_engine):
    get_sqlite_db_location.return_value = "test/app/state.db"
    dao: FineTuningStudioDao = FineTuningStudioDao()
    get_sqlite_db_location.assert_called_once()
    assert create_engine.call_args.kwargs['echo'] == False
    assert create_engine.call_args[0] == "sqlite+pysqlite:///test/app/state.db"


@patch("ft.db.dao.create_engine")
@patch("ft.db.dao.get_sqlite_db_location")
def test_db_dao_init_no_url(get_sqlite_db_location, create_engine):
    dao: FineTuningStudioDao = FineTuningStudioDao("sqlite+pysqlite:///test/other/state.db")
    get_sqlite_db_location.assert_not_called()
    assert create_engine.call_args[0][0] == "sqlite+pysqlite:///test/other/state.db"
    assert create_engine.call_args.kwargs['echo'] == False


class TestDatabaseExports(unittest.TestCase):

    def setUp(self):
        self.test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)
        Base.metadata.create_all(self.test_dao.engine)  # Create tables

    def test_export_no_data(self):
        out_dict: dict = export_to_dict(self.test_dao)
        assert out_dict.keys() == TABLE_TO_MODEL_REGISTRY.keys()
        for _, v in out_dict.items():
            assert v == []

    def test_export_with_data(self):
        with self.test_dao.get_session() as session:
            session.add(Dataset(
                id="ds1",
                type=DatasetType.HUGGINGFACE
            ))

        out_dict: dict = export_to_dict(self.test_dao)
        assert len(out_dict.get("datasets")) == 1
        assert out_dict.get("datasets")[0].get("id") == "ds1"
        assert "description" not in out_dict.get("datasets")[0].keys()


class TestDatabaseImports(unittest.TestCase):

    def setUp(self):
        self.test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)
        Base.metadata.create_all(self.test_dao.engine)  # Create tables

    def test_import_bad_table_name(self):
        import_dict = {
            "some_bad_table_name": [
                {
                    "id": "asdfasdf",
                    "bad_type": "this will not be written to the database."
                }
            ]
        }

        with self.assertRaises(ValueError) as context:
            import_from_dict(import_dict, self.test_dao)

        assert str(context.exception) == "Error importing database from dict: 'some_bad_table_name' is not a valid table name."

    def test_import_happy(self):
        import_dict = {
            "adapters": [
                {
                    "id": "ad1",
                    "model_id": "m1",
                    "type": AdapterType.PROJECT
                }
            ],
            "models": [
                {
                    "id": "m1",
                    "type": ModelType.HUGGINGFACE
                }
            ]
        }

        import_from_dict(import_dict, self.test_dao)

        with self.test_dao.get_session() as session:
            adapters: List[Adapter] = session.query(Adapter).all()
            assert len(adapters) == 1
            assert adapters[0].id == "ad1"
            assert adapters[0].description is None

            models: List[Model] = session.query(Model).all()
            assert len(models) == 1

            adapters_with_m1: List[Adapter] = session.query(Adapter).where(Adapter.model_id == models[0].id).all()
            assert len(adapters_with_m1) == 1
