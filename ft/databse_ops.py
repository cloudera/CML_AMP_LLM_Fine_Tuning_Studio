from ft.db.dao import FineTuningStudioDao
from ft.api import *
from ft.db.db_import_export import DatabaseJsonConverter


def export_database(request: ExportDatabaseRequest,
                    dao: FineTuningStudioDao = None) -> ExportDatabaseResponse:
    db_converter = DatabaseJsonConverter()
    return ExportDatabaseResponse(exported_json=db_converter.export_to_json())


def import_database(request: ImportDatabaseRequest,
                    dao: FineTuningStudioDao = None) -> ImportDatabaseResponse:
    db_converter = DatabaseJsonConverter()
    db_converter.import_from_json(request.imported_json_path)
    return ImportDatabaseResponse()
