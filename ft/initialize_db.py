from ft.db.dao import FineTuningStudioDao
import os
import json
from uuid import uuid4
from typing import Dict
from ft.db.model import Config
from ft.api import *
from sqlalchemy.exc import SQLAlchemyError
from ft.consts import AXOLOTL_DATASET_FORMAT_CONFIGS_FOLDER_PATH


class InitializeDB:
    def __init__(self, dao: FineTuningStudioDao):
        self.dao = dao

    def get_session(self):
        return self.dao.get_session()

    def initialize_all(self):
        self.initialize_axolotl_dataset_type_configs()

    def initialize_axolotl_dataset_type_configs(self):
        config_directory = AXOLOTL_DATASET_FORMAT_CONFIGS_FOLDER_PATH

        # Iterate over all JSON files in the directory
        for filename in os.listdir(config_directory):
            if filename.endswith(".json"):
                file_path = os.path.join(config_directory, filename)

                # Read and unmarshal the JSON file
                with open(file_path, 'r') as f:
                    try:
                        data_dict = json.load(f)
                        dataset_formats_collection = DatasetFormatsCollection(**data_dict)
                    except Exception as e:
                        print(f"Error parsing {filename}: {e}")
                        continue

                # Process each dataset type in the unmarshalled data
                for dataset_key, dataset_info in dataset_formats_collection.dataset_formats.items():
                    self._add_config(
                        config_type=ConfigType.AXOLOTL_DATASET_FORMATS,
                        description=dataset_info.name,
                        config_content=dataset_info.format  # Use the format field from the unmarshalled model
                    )

    def _add_config(self, config_type: ConfigType, description: str, config_content: Dict):
        # Validate inputs
        if not description:
            raise ValueError("Description cannot be empty.")

        if not isinstance(config_content, dict) or not config_content:
            raise ValueError("Config content must be a non-empty dictionary.")

        try:
            with self.dao.get_session() as session:
                # Convert the config_content dict to a JSON string
                config_content_str = json.dumps(config_content)

                # Query existing configs of the same type
                configs = session.query(Config).filter(Config.type == config_type).all()

                # Check for existing similar configs
                similar_configs = list(
                    filter(
                        lambda x: json.loads(x.config) == config_content,
                        configs
                    )
                )

                # Ensure at most one similar config exists
                if len(similar_configs) > 1:
                    raise RuntimeError("Multiple similar configs found. Manual cleanup may be required.")

                # If a similar config exists, delete the existing one
                if len(similar_configs) == 1:
                    existing_config = similar_configs[0]
                    session.delete(existing_config)

                # Add the new config
                new_config = Config(
                    id=str(uuid4()),
                    type=config_type,
                    description=description,
                    config=config_content_str
                )
                session.add(new_config)

        except SQLAlchemyError as e:
            raise RuntimeError(f"Database error occurred: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error in JSON handling: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred: {e}")
