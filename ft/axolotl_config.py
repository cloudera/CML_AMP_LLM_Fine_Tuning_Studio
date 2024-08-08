import yaml
from typing import Optional
from google.protobuf.json_format import MessageToDict, ParseDict
from ft.api import *

# Assuming you've imported the AxolotlTrainConfig and other necessary Protobuf classes


class AxolotlConfig:
    def __init__(self, config: Optional[AxolotlTrainConfig] = None):
        self.config = config if config is not None else AxolotlTrainConfig()

    # Method to set a nested value in the config using dot notation
    def set_value(self, key: str, value):
        keys = key.split('.')
        current = self.config
        try:
            for k in keys[:-1]:
                current = getattr(current, k)
            setattr(current, keys[-1], value)
        except AttributeError as e:
            raise ValueError(f"Invalid configuration key: {key}") from e

    # Convert the Protobuf config to a dictionary
    def to_dict(self):
        data_dict = MessageToDict(self.config, preserving_proto_field_name=True)
        return self._include_special_values(data_dict, self.config)

    # Save the configuration to a YAML file
    def save_to_yaml(self, file_path: str):
        data_dict = self.to_dict()
        print("Keys with empty strings, False, or None:")
        self._print_special_keys(data_dict)
        try:
            with open(file_path, 'w') as file:
                yaml.dump(data_dict, file, default_flow_style=False, sort_keys=False)
        except IOError as e:
            raise IOError(f"Unable to save to file: {file_path}") from e

    # Load the configuration from a YAML file
    @classmethod
    def load_from_yaml(cls, file_path: str):
        try:
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)
        except IOError as e:
            raise IOError(f"Unable to load from file: {file_path}") from e
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML file: {file_path}") from e

        config = AxolotlTrainConfig()
        ParseDict(data, config)
        return cls(config)

    # Get the current configuration object
    def get_config(self):
        return self.config

    # Recursively include special values (None, '', False) in the dictionary
    def _include_special_values(self, data: dict, proto):
        if isinstance(data, dict):
            for field in proto.DESCRIPTOR.fields:
                field_name = field.name
                if field_name not in data:
                    value = getattr(proto, field_name)
                    if value in [None, '', False]:
                        data[field_name] = value
            for k, v in data.items():
                if isinstance(v, dict):
                    field_proto = getattr(proto, k)
                    data[k] = self._include_special_values(v, field_proto)
        return data

    # Print keys with special values (None, '', False)
    def _print_special_keys(self, data: dict, parent_key=''):
        if isinstance(data, dict):
            for k, v in data.items():
                full_key = f"{parent_key}.{k}" if parent_key else k
                if v in [None, '', False]:
                    print(f"{full_key}: {v}")
                if isinstance(v, dict):
                    self._print_special_keys(v, full_key)

# Example usage with edge case handling
# try:
#     config = AxolotlConfig()
#     config.set_value('base_model', 'NousResearch/Llama-2-7b-hf')
#     config.set_value('learning_rate', 0.0002)
#     config.save_to_yaml('axolotl_config.yaml')

#     loaded_config = AxolotlConfig.load_from_yaml('axolotl_config.yaml')
#     print(loaded_config.to_dict())
# except (ValueError, IOError) as e:
#     print(f"Error: {e}")
