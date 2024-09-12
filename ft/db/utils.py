
from ft.db.model import *
from ft.db.dao import FineTuningStudioDao
from ft.consts import DEFAULT_PROJECT_DEFAULTS_LOCATION
from typing import List
import json

import os


def get_project_defaults_location():
    """
    Get the location of the currently loaded state file.
    """
    if os.environ.get("FINE_TUNING_STUDIO_PROJECT_DEFAULTS"):
        return os.environ.get("FINE_TUNING_STUDIO_PROJECT_DEFAULTS")
    return DEFAULT_PROJECT_DEFAULTS_LOCATION


def export_to_dict(dao: FineTuningStudioDao = None) -> None:
    """
    Export the current database to a JSON

    todo: confirm null/None behavior
    """
    table_classes: List[MappedDict] = MODEL_TO_TABLE_REGISTRY.keys()
    output_json = {}
    with dao.get_session() as session:
        for cls in table_classes:
            table_name = MODEL_TO_TABLE_REGISTRY.get(cls)
            table_rows: List[MappedDict] = session.query(cls).all()
            output_json[table_name] = [row.to_dict() for row in table_rows]
    return output_json


def import_from_dict(db_dict: dict, dao: FineTuningStudioDao = None) -> None:
    """
    Import data from a dictionary into the database. Data must take the form of a
    JSON dict where each key is the table name, and each value is a list of dicts
    that represent the declarative base model table row.

    todo: confirm null/None behavior
    """

    with dao.get_session() as session:
        for table_name, table_rows in db_dict.items():
            if table_name not in TABLE_TO_MODEL_REGISTRY.keys():
                raise ValueError(f"Error importing database from dict: '{table_name}' is not a valid table name.")
            table_cls = TABLE_TO_MODEL_REGISTRY.get(table_name)
            for table_row_dict in table_rows:
                table_row = table_cls(**table_row_dict)
                session.add(table_row)  # todo: determine behavior if duplicate data is added
    return


def import_defaults():
    """
    Import project defaults into the Studio.
    """

    # Get the project defaults
    defaults_file = get_project_defaults_location()

    # Create a brand new DAO (and a brand new .app/state.db) and write project defaults.
    dao: FineTuningStudioDao = FineTuningStudioDao()
    import_dict = json.load(open(defaults_file, 'r'))
    import_from_dict(import_dict, dao=dao)
