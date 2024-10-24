import unittest
from unittest.mock import patch, mock_open
from pydantic import ValidationError
from ft.db.dao import FineTuningStudioDao
from ft.db.model import Base, Config
from ft.api import DatasetFormatsCollection, DatasetFormatInfo
from ft.initialize_db import InitializeDB
from ft.api import ConfigType
import json
from sqlalchemy.exc import SQLAlchemyError


class TestInitializeDB(unittest.TestCase):

    def setUp(self):
        # Set up an in-memory SQLite database
        self.test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)
        Base.metadata.create_all(self.test_dao.engine)  # Create tables

    @patch('os.listdir', return_value=[])
    @patch('builtins.open', new_callable=mock_open, read_data='{}')
    def test_initialize_axolotl_dataset_type_configs_empty_directory(self, mock_open, mock_listdir):
        init_db = InitializeDB(self.test_dao)

        init_db.initialize_axolotl_dataset_type_configs()

        # Verify no configs were added since the directory is empty
        with self.test_dao.get_session() as session:
            self.assertEqual(session.query(Config).count(), 0)

    @patch('os.listdir', return_value=['test_config.json'])
    @patch('builtins.open', new_callable=mock_open, read_data='{}')
    @patch('json.load')
    def test_initialize_axolotl_dataset_type_configs(self, mock_json_load, mock_open, mock_listdir):
        # Prepare the mocked data
        mocked_data_dict = {
            "dataset_formats": {
                "dummy_dataset": {
                    "name": "Dummy Dataset",
                    "description": "A dummy dataset format",
                    "format": {"key": "value"}
                }
            }
        }

        # Mock json.load to return the above dictionary
        mock_json_load.return_value = mocked_data_dict

        # Create DatasetFormatInfo instances from the mocked data
        dataset_format_info = DatasetFormatInfo(
            name="Dummy Dataset",
            description="A dummy dataset format",
            format={"key": "value"}
        )

        # Mock DatasetFormatsCollection to return a proper instance
        dataset_formats_collection = DatasetFormatsCollection(
            dataset_formats={"dummy_dataset": dataset_format_info}
        )

        with patch('ft.api.DatasetFormatsCollection', return_value=dataset_formats_collection):
            init_db = InitializeDB(self.test_dao)

            init_db.initialize_axolotl_dataset_type_configs()

            # Verify that the config was added correctly
            with self.test_dao.get_session() as session:
                config = session.query(Config).filter_by(description="Dummy Dataset").one()
                self.assertEqual(config.config, json.dumps({"key": "value"}))

    @patch('os.listdir', return_value=['test_config.json'])
    @patch('builtins.open', new_callable=mock_open, read_data='{"invalid": "data"}')
    def test_initialize_axolotl_dataset_type_configs_validation_error(self, mock_open, mock_listdir):
        # Define a function that raises ValidationError
        def raise_validation_error(*args, **kwargs):
            raise ValidationError(
                [{'loc': ('dataset_formats', 'dummy_dataset'), 'msg': 'Invalid format', 'type': 'type_error'}],
                DatasetFormatsCollection
            )

        # Patch DatasetFormatsCollection's initialization to raise ValidationError
        with patch.object(DatasetFormatsCollection, '__init__', side_effect=raise_validation_error):
            init_db = InitializeDB(self.test_dao)

            init_db.initialize_axolotl_dataset_type_configs()

            # Verify no configs were added due to the validation error
            with self.test_dao.get_session() as session:
                self.assertEqual(session.query(Config).count(), 0)

    def test_add_config_success(self):
        init_db = InitializeDB(self.test_dao)

        init_db._add_config(
            config_type=ConfigType.AXOLOTL_DATASET_FORMATS,
            description="Test Config",
            config_content={"key": "value"}
        )

        # Verify the config was added successfully
        with self.test_dao.get_session() as session:
            config = session.query(Config).filter_by(description="Test Config").one()
            self.assertEqual(config.config, json.dumps({"key": "value"}))

    def test_add_config_existing(self):
        init_db = InitializeDB(self.test_dao)

        # Add an existing config first
        with self.test_dao.get_session() as session:
            config = Config(
                id="existing_id",
                type=ConfigType.AXOLOTL_DATASET_FORMATS,
                description="Existing Config",
                config=json.dumps({"key": "value"})
            )
            session.add(config)
            session.commit()

        # Now attempt to add a similar config
        init_db._add_config(
            config_type=ConfigType.AXOLOTL_DATASET_FORMATS,
            description="Existing Config",
            config_content={"key": "value"}
        )

        # Verify the config was not duplicated
        with self.test_dao.get_session() as session:
            configs = session.query(Config).filter_by(description="Existing Config").all()
            self.assertEqual(len(configs), 1)

    def test_add_config_error(self):
        init_db = InitializeDB(self.test_dao)

        # Simulate a SQLAlchemy error during the add operation
        with patch.object(FineTuningStudioDao, 'get_session', side_effect=SQLAlchemyError("Test SQLAlchemy error")):
            with self.assertRaises(RuntimeError) as context:
                init_db._add_config(
                    config_type=ConfigType.AXOLOTL_DATASET_FORMATS,
                    description="Test Config",
                    config_content={"key": "value"}
                )
            self.assertIn("Database error occurred", str(context.exception))
