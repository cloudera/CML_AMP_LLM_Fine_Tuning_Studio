
import yaml
from ft.api import *
import json

from ft.db.dao import FineTuningStudioDao
from ft.db.model import Config, Model
import sqlalchemy
from sqlalchemy import delete
from sqlalchemy.orm.session import Session

from typing import List
from ft import consts
from uuid import uuid4
from ft.consts import *
from ft.utils import dict_to_yaml_string
from ft.config.model_configs.config_loader import ModelMetadataFinder



def get_configs_for_model_id(session: Session, configs: List[Config], model_id: str) -> List[Config]:

    # Get the model type.
    model: Model = session.get(Model, model_id)
    # TODO: implement extracting the best config based on model data.
    model_hf_name = model.huggingface_model_name
    model_metadata_finder = ModelMetadataFinder(model_hf_name)
    model_family = model_metadata_finder.fetch_model_family_from_config()
    # Filter configs for default configs
    model_family = model_family + DEFAULT_CONFIG_DESCRIPTION
    filtered_configs = [c for c in configs if c.description == model_family]
    # if filtered_configs == 0, show default config
    if len(filtered_configs) == 0:
        return configs 
    return filtered_configs

def transform_name_to_family(model_name: str) -> str:
    model_metadata_finder = ModelMetadataFinder(model_name)
    model_family = model_metadata_finder.fetch_model_family_from_config()
    return model_family + USER_CONFIG_DESCRIPTION

def list_configs(request: ListConfigsRequest, dao: FineTuningStudioDao = None) -> ListConfigsResponse:

    response: ListConfigsResponse = ListConfigsResponse()

    # Start up a DB session.
    with dao.get_session() as session:

        # Determine if config type is part of the request, which will be used
        # for an extra layer of filtering.
        if 'type' in [x[0].name for x in request.ListFields()]:
            configs: List[Config] = session.query(Config).where(Config.type == request.type).all()
        else:
            configs: List[Config] = session.query(Config).all()

        # TODO: determine logic for listing configs based on
        # model id, adapter id, etc.
        if 'model_id' in [x[0].name for x in request.ListFields()]:
            configs: List[Config] = get_configs_for_model_id(session, configs, request.model_id)

        # Add final configs to the response message.
        response.configs.extend(list(map(lambda x: x.to_protobuf(ConfigMetadata), configs)))

    return response


def get_config(request: GetConfigRequest, dao: FineTuningStudioDao = None) -> GetConfigResponse:
    response = GetConfigResponse()

    with dao.get_session() as session:
        config: Config = session.query(Config).where(Config.id == request.id).one()
        response = GetConfigResponse(
            config=config.to_protobuf(ConfigMetadata) if config is not None else None
        )

    return response


# Need more discussion about this
'''
def get_config(request: GetConfigRequest, dao: FineTuningStudioDao = None) -> GetConfigResponse:
    response = GetConfigResponse()
    try:
        if request.type == None:
            with dao.get_session() as session:
                config: Config = session.query(Config).where(Config.id == request.id).one()
                response = GetConfigResponse(
                    config=config.to_protobuf(ConfigMetadata) if config is not None else None
                )
        else:
            with dao.get_session() as session:
                try:
                    config: Config = session.query(Config).where(Config.type == request.type, Config.description == request.description).one()
                    response = GetConfigResponse(
                        config=config.to_protobuf(ConfigMetadata) if config is not None else None
                    )
                # get default configs if the config for the model family is not present
                except sqlalchemy.orm.exc.NoResultFound:
                    config: Config = session.query(Config).where(Config.type == request.type, Config.description == consts.DEFAULT_CONFIG_DESCRIPTION).one()
                    response = GetConfigResponse(
                        config=config.to_protobuf(ConfigMetadata) if config is not None else None
                    )
        return response
    except Exception as e:
        raise ValueError(f"ERROR: Failed to get config. {e}")
'''

def add_config(request: AddConfigRequest, dao: FineTuningStudioDao = None) -> AddConfigResponse:
    """
    Add a new configuration to the datastore. Returns a configuration metadata object
    with a configuration id. The configuration store for adding new configs acts as a
    cache. If there is an identical config of the same type with the same config
    internals, then the existing configuration and id are passed back out of this
    request. If there is no preexisting config that matches the config request, a new
    config is added.
    """

    response: AddConfigResponse = AddConfigResponse()

    with dao.get_session() as session:
        if 'description' in [x[0].name for x in request.ListFields()]:
            description = transform_name_to_family(request.description)
            configs: List[Config] = session.query(Config).where(Config.type == request.type, Config.description == description).all()
        else:
            configs: List[Config] = session.query(Config).where(Config.type == request.type).all()

        # Handle AXOLOTL type by parsing the config as YAML
        if request.type == ConfigType.AXOLOTL:
            request_config_dict = yaml.safe_load(request.config)
            # Convert the dict back to a YAML string
            config_content = dict_to_yaml_string(request_config_dict)
        else:
            request_config_dict = json.loads(request.config)
            # Convert the dict back to a JSON string
            config_content = json.dumps(request_config_dict)

        # Ensure that there are no similar configs by comparing the parsed config
        similar_configs: List[Config] = list(
            filter(
                lambda x: (
                    yaml.safe_load(x.config) if request.type == ConfigType.AXOLOTL else json.loads(x.config)
                ) == request_config_dict,
                configs
            )
        )

        # Ensure that we only have at most one similar config for a given type.
        assert len(similar_configs) <= 1

        # If we found a pre-existing config, then return that config ID.
        if len(similar_configs) == 1:
            response = AddConfigResponse(
                config=similar_configs[0].to_protobuf(ConfigMetadata)
            )
        else:
            # If no similar config exists, add the new config.
            config: Config = Config(
                id=str(uuid4()),
                type=request.type,
                description=request.description,
                config=config_content
            )
            session.add(config)
            response = AddConfigResponse(
                config=config.to_protobuf(ConfigMetadata)
            )

    return response


def remove_config(request: RemoveConfigRequest, dao: FineTuningStudioDao = None) -> RemoveConfigResponse:
    with dao.get_session() as session:
        session.execute(delete(Config).where(Config.id == request.id))
    return RemoveConfigResponse()
