import pytest
import sqlite3
import json
from ft.db.db_import_export import DatabaseJsonConverter

# Fixtures provide a cleaner way to set up test dependencies
# They can be reused across multiple tests and are managed automatically by pytest


@pytest.fixture
def temp_db_path(tmp_path):
    """
    Creates a temporary database path using pytest's tmp_path fixture.
    This is automatically cleaned up after tests complete.
    """
    return tmp_path / "test.db"


@pytest.fixture
def temp_json_path(tmp_path):
    """
    Creates a temporary JSON file path using pytest's tmp_path fixture.
    """
    return tmp_path / "test.json"


@pytest.fixture
def db_converter(temp_db_path):
    """
    Creates a DatabaseJsonConverter instance with the temporary database.
    """
    return DatabaseJsonConverter(str(temp_db_path))


@pytest.fixture
def sample_database(temp_db_path):
    """
    Creates a sample database with test data.
    Returns the database path for use in tests.
    """
    with sqlite3.connect(temp_db_path) as conn:
        cursor = conn.cursor()

        # Create products table
        cursor.execute("""
            CREATE TABLE models (\n\tid VARCHAR NOT NULL, \n\ttype VARCHAR, \n\tframework VARCHAR, \n\tname VARCHAR, \n\tdescription VARCHAR, \n\thuggingface_model_name VARCHAR, \n\tlocation VARCHAR, \n\tcml_registered_model_id VARCHAR, \n\tmlflow_experiment_id VARCHAR, \n\tmlflow_run_id VARCHAR, \n\tPRIMARY KEY (id)\n)
        """)

        # Insert sample products
        models = [
            ("f1c3d635-980d-4114-a43e-b3a2eba13910", "bigscience/bloom-1b1")
        ]
        cursor.executemany(
            "INSERT INTO models (id, name) VALUES (?, ?)",
            models
        )

        conn.commit()

    return temp_db_path


@pytest.fixture
def exported_json(db_converter, sample_database, temp_json_path):
    """
    Exports the sample database to JSON and returns the JSON path.
    This fixture depends on other fixtures, demonstrating pytest's automatic dependency resolution.
    """
    db_converter.export_to_json(temp_json_path)
    return temp_json_path


def test_export_creates_file(db_converter, sample_database, temp_json_path):
    """
    Test that export operation creates a JSON file.
    """
    db_converter.export_to_json(temp_json_path)
    assert temp_json_path.exists(), "JSON file was not created"


def test_export_content(exported_json):
    """
    Test that exported JSON contains correct data structure and content.
    Uses the exported_json fixture which provides a pre-exported JSON file.
    """
    with open(exported_json) as f:
        data = json.load(f)

    # Verify database structure
    assert "models" in data, "Models table missing from export"

    # Verify table schemas
    assert "schema" in data["models"], "Models schema missing"

    # Verify data content
    models = data["models"]["data"]
    assert len(models) == 1, "Wrong number of models"
    assert models[0]["id"] == "f1c3d635-980d-4114-a43e-b3a2eba13910", "models id mismatch"
    assert models[0]["name"] == "bigscience/bloom-1b1", "models name mismatch"


def test_import_to_new_database(exported_json, tmp_path):
    """
    Test importing JSON into a new database.
    """
    # Create a new database path
    new_db_path = tmp_path / "new.db"

    # Create new converter and import data
    new_converter = DatabaseJsonConverter(str(new_db_path))
    new_converter.import_from_json(exported_json)

    # Verify imported data
    with sqlite3.connect(new_db_path) as conn:
        cursor = conn.cursor()

        # Check products table
        cursor.execute("SELECT COUNT(*) FROM models")
        assert cursor.fetchone()[0] == 1, "Wrong number of models imported"

        cursor.execute("SELECT id, name FROM models")
        models = cursor.fetchone()
        assert models == ("f1c3d635-980d-4114-a43e-b3a2eba13910",
                          "bigscience/bloom-1b1"), "Product data mismatch after import"


def test_nonexistent_json_import(db_converter):
    """
    Test handling of importing non-existent JSON file.
    """
    with pytest.raises(FileNotFoundError):
        db_converter.import_from_json("nonexistent.json")


def test_invalid_json_import(db_converter, temp_json_path):
    """
    Test handling of invalid JSON content.
    """
    # Create invalid JSON file
    temp_json_path.write_text("This is not valid JSON")

    with pytest.raises(json.JSONDecodeError):
        db_converter.import_from_json(temp_json_path)


def test_invalid_database_path():
    """
    Test handling of invalid database path.
    """
    invalid_converter = DatabaseJsonConverter("/nonexistent/path/db.sqlite")

    with pytest.raises(Exception):
        invalid_converter.export_to_json("test.json")


@pytest.mark.parametrize("table_name,expected_count", [
    ("models", 1)
])
def test_table_record_counts(sample_database, table_name, expected_count):
    """
    Parametrized test to verify record counts in different tables.
    Demonstrates pytest's parametrize feature for running the same test with different inputs.
    """
    with sqlite3.connect(sample_database) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        assert count == expected_count, f"Wrong number of records in {table_name}"
