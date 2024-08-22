
from ft.api import *
import json

from ft.db.dao import FineTuningStudioDao
from ft.db.model import Config, Model

from sqlalchemy import delete
from sqlalchemy.orm.session import Session

from typing import List

from uuid import uuid4


def get_configs_for_model_id(session: Session, configs: List[Config], model_id: str) -> List[Config]:

    # Get the model type.
    model: Model = session.get(Model, model_id)

    # TODO: implement extracting the best config based on model data.
    model_hf_name = model.huggingface_model_name

    return configs


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
        configs: List[Config] = session.query(Config).where(Config.type == request.type).all()

        # ensure that there are no similar configs.
        # we load to a dictionary to do a fully formatted comparison.
        similar_configs: List[Config] = list(
            filter(
                lambda x: json.loads(
                    x.config) == json.loads(
                    request.config),
                configs))

        # ensure that we only have at least one similar config for a given type.
        # if we have more, then we messed up our caching mechanism somwehere
        # along the way, and we are in a bad state (for now!).
        assert len(similar_configs) <= 1

        # If we found a pre-existing config, then return
        # that config ID. If not, then just add the config.
        if len(similar_configs) == 1:
            response = AddConfigResponse(
                config=similar_configs[0].to_protobuf(ConfigMetadata)
            )
        else:
            config: Config = Config(
                id=str(uuid4()),
                type=request.type,
                description=request.description,
                config=json.dumps(json.loads(request.config))
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
