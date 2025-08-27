import sqlite3
import json
import logging
from pathlib import Path
import re
from ft.db.dao import get_sqlite_db_location


class DatabaseJsonConverter:
    """
    A utility class for converting SQLite databases to JSON and vice versa.
    Handles both export of existing databases and import from JSON files.
    """

    def __init__(self, db_path=None):
        """
        Initialize the converter with a database path.

        Args:
            db_path (str): Path to the SQLite database file
        """
        if db_path is None:
            self.db_path = get_sqlite_db_location()
        else:
            self.db_path = db_path

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _get_table_schema(self, cursor: sqlite3.Cursor, table_name: str) -> str:
        """
        Retrieve the CREATE TABLE statement for a given table.

        Args:
            cursor (sqlite3.Cursor): Database cursor
            table_name (str): Name of the table

        Returns:
            str: CREATE TABLE statement
        """
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        return cursor.fetchone()[0]

    def export_to_json(self, output_path=None) -> str:
        """
        Export the entire database structure and content to a JSON file.

        Args:
            output_path (str): Path where the JSON file will be saved
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get all table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                tables = cursor.fetchall()

                database_dict = {}

                for (table_name,) in tables:
                    self.logger.info(f"Exporting table: {table_name}")

                    # Get table schema
                    schema = self._get_table_schema(cursor, table_name)
                    pattern = r'\bCREATE\s+TABLE\b(?!\s+IF\s+NOT\s+EXISTS\b)\s*'
                    replacement = 'CREATE TABLE IF NOT EXISTS '
                    schema = re.sub(pattern, replacement, schema, flags=re.IGNORECASE)
                    # Get table data
                    cursor.execute(f"SELECT * FROM {table_name}")
                    columns = [description[0] for description in cursor.description]
                    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

                    database_dict[table_name] = {
                        "schema": schema,
                        "data": rows
                    }

                # # Write to JSON file
                if output_path is not None:
                    with open(output_path, 'w') as f:
                        json.dump(database_dict, f, indent=2)

                self.logger.info(f"Successfully exported database.")
                return json.dumps(database_dict, indent=2)
        except sqlite3.Error as e:
            self.logger.error(f"SQLite error occurred: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error occurred during export: {e}")
            raise

    def import_from_json(self, json_path: str) -> None:
        """
        Import database structure and content from a JSON file.

        Args:
            json_path (str): Path to the JSON file containing database structure and data
        """
        try:
            # Validate JSON file exists
            if not Path(json_path).exists():
                raise FileNotFoundError(f"JSON file not found: {json_path}")

            with open(json_path, 'r') as f:
                database_dict = json.load(f)

            # Validate JSON structure
            if not isinstance(database_dict, dict):
                raise ValueError("Invalid JSON structure: root must be an object")

            # Create new database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for table_name, table_info in database_dict.items():
                    self.logger.info(f"Importing table: {table_name}")

                    # Validate table information
                    if not isinstance(table_info, dict):
                        raise ValueError(f"Invalid table information for table {table_name}")
                    if "schema" not in table_info or "data" not in table_info:
                        raise ValueError(f"Missing schema or data for table {table_name}")

                    # Create table
                    # This is ommited now as we are assuming migrations have already run
                    try:
                        cursor.execute(table_info["schema"])
                    except sqlite3.Error as e:
                        self.logger.error(f"Error creating table {table_name}: {e}")
                        raise

                    # Insert data
                    if table_info["data"]:
                        # Get column names from first row
                        columns = list(table_info["data"][0].keys())
                        placeholders = ','.join(['?' for _ in columns])
                        insert_sql = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})"
                        not_inserted_rows = []
                        try:
                            for row in table_info["data"]:
                                # Validate row data
                                if not all(col in row for col in columns):
                                    raise ValueError(f"Missing columns in data row for table {table_name}")
                                cursor.execute(insert_sql, [row[col] for col in columns])
                        except sqlite3.Error as e:
                            self.logger.error(f"Error inserting data into table {table_name}: {e}")
                            not_inserted_rows.append(row)

                        # Log not inserted rows
                        if not_inserted_rows:
                            self.logger.warning(f"Not inserted data into table {table_name}:")
                            for row in not_inserted_rows:
                                self.logger.warning(f"{row}")

                conn.commit()
                self.logger.info(f"Successfully imported database from {json_path}")

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON file: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error occurred during import: {e}")
            raise


def main():
    """
    Example usage of the DatabaseJsonConverter class.
    """
    # Example usage
    converter = DatabaseJsonConverter("/Users/abhishek.ranjan/Work/amp/fine-tuning-studio/.app/state.db")

    # Export database to JSON
    # try:
    #     converter.export_to_json("database_backup.json")
    # except Exception as e:
    #     print(f"Export failed: {e}")

    # Import database from JSON
    try:
        converter.import_from_json("database_backup.json")
    except Exception as e:
        print(f"Import failed: {e}")


if __name__ == "__main__":
    main()
