from ft.api import *
from ft.state import write_state, replace_state_field
import json
import yaml
from uuid import uuid4

# Custom representer to handle empty strings
def represent_none(self, _):
    return self.represent_scalar('tag:yaml.org,2002:null', '')

# Register the custom representer for NoneType
yaml.add_representer(type(None), represent_none)

def list_configs(state: AppState, request: ListConfigsRequest) -> ListConfigsResponse:
    configs = state.configs
    if request.type != ConfigType.CONFIG_TYPE_UNKNOWN:
        configs = list(filter(lambda x: x.type == request.type, configs))
    return ListConfigsResponse(
        configs=configs
    )


def get_config(state: AppState, request: GetConfigRequest) -> GetConfigResponse:
    configs = list(filter(lambda x: x.id == request.id, state.configs))
    assert len(configs) == 1
    return GetConfigResponse(
        config=configs[0]
    )


def add_config(state: AppState, request: AddConfigRequest) -> AddConfigResponse:
    """
    Add a new configuration to the datastore. Returns a configuration metadata object
    with a configuration id. The configuration store for adding new configs acts as a
    cache. If there is an identical config of the same type with the same config
    internals, then the existing configuration and id are passed back out of this
    request. If there is no preexisting config that matches the config request, a new
    config is added.
    """

    # Collect a list of configurations by type
    configs = list(filter(lambda x: x.type == request.type, state.configs))

    # Handle CONFIG_TYPE_AXOLOTL_TRAIN_CONFIG specifically as YAML
    if request.type == ConfigType.CONFIG_TYPE_AXOLOTL_TRAIN_CONFIG:
        try:
            request_config_yaml = yaml.safe_load(request.config)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")

        # Check for an existing identical YAML config
        similar_configs = list(
            filter(lambda x: yaml.safe_load(x.config) == request_config_yaml, configs)
        )

        if len(similar_configs) == 1:
            return AddConfigResponse(
                config=similar_configs[0]
            )
        else:
            new_config: ConfigMetadata = ConfigMetadata(
                id=str(uuid4()),
                type=request.type,
                description=request.description,
                config=yaml.dump(request_config_yaml, default_flow_style=False)  # Store the formatted YAML
            )
            state.configs.append(new_config)
            write_state(state)
            return AddConfigResponse(
                config=new_config
            )

    # Handle other config types as JSON
    else:
        # If there are no configs of this type yet, then add it!
        if len(configs) == 0:
            new_config: ConfigMetadata = ConfigMetadata(
                id=uuid4(),
                type=request.type,
                description=request.description,
                config=request.config
            )
            state.configs.append(new_config)
            write_state(state)
            return AddConfigResponse(
                config=new_config
            )

        # ensure that there are no similar configs.
        # we load to a dictionary to do a fully formatted comparison.
        similar_configs = list(filter(lambda x: json.loads(x.config) == json.loads(request.config), configs))

        # ensure that we only have at least one similar config for a given type.
        # if we have more, then we messed up our caching mechanism somwehere
        # along the way, and we are in a bad state (for now!).
        assert len(similar_configs) <= 1

        # If we found a pre-existing config, then return
        # that config ID. If not, then just add the config.
        if len(similar_configs) == 1:
            return AddConfigResponse(
                config=similar_configs[0]
            )
        else:
            new_config: ConfigMetadata = ConfigMetadata(
                id=str(uuid4()),
                type=request.type,
                description=request.description,
                config=json.dumps(json.loads(request.config))  # Fix formatting
            )
            state.configs.append(new_config)
            write_state(state)
            return AddConfigResponse(
                config=new_config
            )


def remove_config(state: AppState, request: RemoveConfigRequest) -> RemoveConfigResponse:
    configs = list(filter(lambda x: not x.id == request.id, state.configs))
    state = replace_state_field(state, configs=configs)
    return RemoveConfigResponse()
