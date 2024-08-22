from unittest.mock import patch

from ft.db.dao import get_sqlite_db_location, FineTuningStudioDao, DEFAULT_SQLITE_DB_LOCATION


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
    assert create_engine.call_args.kwargs['echo']
    assert create_engine.call_args[0] == "sqlite+pysqlite:///test/app/state.db"


@patch("ft.db.dao.create_engine")
@patch("ft.db.dao.get_sqlite_db_location")
def test_db_dao_init_no_url(get_sqlite_db_location, create_engine):
    dao: FineTuningStudioDao = FineTuningStudioDao("sqlite+pysqlite:///test/other/state.db")
    get_sqlite_db_location.assert_not_called()
    assert create_engine.call_args[0][0] == "sqlite+pysqlite:///test/other/state.db"
    assert create_engine.call_args.kwargs['echo']
