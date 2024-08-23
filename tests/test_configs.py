
import pytest

from ft.api import *

from unittest.mock import patch

from ft.configs import (
    get_config,
    list_configs,
    add_config,
    remove_config,
)

from ft.db.dao import FineTuningStudioDao
from ft.db.model import Config

from sqlalchemy.exc import NoResultFound
from typing import List

import json

from ft.utils import dict_to_yaml_string


def test_get_config_happy():

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Config(id="c1", config=json.dumps({"k1": "v1"})))
        session.add(Config(id="c2", config=json.dumps({"k2": "v2"})))

    out: ConfigMetadata = get_config(GetConfigRequest(id="c2"), dao=test_dao).config
    assert out.id == "c2"
    assert out.config == '{"k2": "v2"}'


def test_get_config_missing():

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Config(id="c1", config=json.dumps({"k1": "v1"})))
        session.add(Config(id="c2", config=json.dumps({"k2": "v2"})))

    with pytest.raises(NoResultFound):
        out: ConfigMetadata = get_config(GetConfigRequest(id="c3"), dao=test_dao).config


def test_list_state_configs_happy():

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Config(id="c1", config=json.dumps({"k1": "v1"})))
        session.add(Config(id="c2", config=json.dumps({"k2": "v2"})))

    configs = list_configs(ListConfigsRequest(), dao=test_dao).configs
    assert len(configs) == 2


def test_list_state_configs_by_type():

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Config(id="c1", type=ConfigType.TRAINING_ARGUMENTS, config=json.dumps({"k1": "v1"})))
        session.add(Config(id="c2", type=ConfigType.BITSANDBYTES_CONFIG, config=json.dumps({"k2": "v2"})))

    configs: List[ConfigMetadata] = list_configs(ListConfigsRequest(
        type=ConfigType.TRAINING_ARGUMENTS), dao=test_dao).configs
    assert len(configs) == 1
    assert configs[0].config == '{"k1": "v1"}'


@patch("ft.configs.uuid4")
def test_add_config_no_prior_configs(uuid4):
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    test_config = json.dumps({"load_in_8bit": True})

    uuid4.return_value = "c1"

    response_config: ConfigMetadata = add_config(AddConfigRequest(
        type=ConfigType.BITSANDBYTES_CONFIG,
        config=test_config
    ), dao=test_dao).config

    assert isinstance(response_config, ConfigMetadata)
    with test_dao.get_session() as session:
        assert session.get(Config, "c1").config == '{"load_in_8bit": true}'


@patch("ft.configs.uuid4")
def test_add_axolotl_config_no_prior_configs(uuid4):
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    test_config = dict_to_yaml_string({"base_model": "tinyllama", "flash_attention": None})

    uuid4.return_value = "c1"

    response_config: ConfigMetadata = add_config(AddConfigRequest(
        type=ConfigType.AXOLOTL,
        config=test_config
    ), dao=test_dao).config

    assert isinstance(response_config, ConfigMetadata)
    with test_dao.get_session() as session:
        assert session.get(Config, "c1").config == 'base_model: tinyllama\nflash_attention:\n'


@patch("ft.configs.uuid4")
def test_add_config_existing_configs(uuid4):
    """
    Tests to make sure that two identical configs do not
    get stored into the config store. Note that formatting
    of the config string does not matter so long as the content is
    identical between the two configs.
    """

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Config(id="c1", type=ConfigType.TRAINING_ARGUMENTS, config=json.dumps({"k1": "v1"})))

    response_config: ConfigMetadata = add_config(
        AddConfigRequest(
            type=ConfigType.TRAINING_ARGUMENTS,
            config='\n   { "k1"    : "v1"\n\n\n}'
        ),
        dao=test_dao
    ).config

    assert response_config.id == "c1"
    uuid4.assert_not_called()


def test_add_config_caching_mechanism_broke():
    """
    Right now the config stores are designed as "cahcing" mechanism of
    sorts for basic configs (generation configs, quantization configs, etc.), which
    means that two identical configs should not exist in the datastore with different
    IDs. This might be allowed in the future (and is in fact allowed for other metadata
    types already like datasets, etc.), but given the caching type implementation of
    the add_config() feature for now, we'd like to mark this as an "antipattern"
    and discourage it (for now!).
    """

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Config(id="c1", type=ConfigType.TRAINING_ARGUMENTS, config=json.dumps({"k1": "v1"})))
        session.add(Config(id="c2", type=ConfigType.TRAINING_ARGUMENTS, config=json.dumps({"k1": "v1"})))

    with pytest.raises(AssertionError):
        response: AddConfigResponse = add_config(
            AddConfigRequest(
                type=ConfigType.TRAINING_ARGUMENTS,
                config=json.dumps({"k1": "v1"})
            ),
            dao=test_dao
        )


@patch("ft.configs.uuid4")
def test_add_config_existing_unique_configs(uuid4):
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Config(id="c1", type=ConfigType.TRAINING_ARGUMENTS, config=json.dumps({"k1": "v1"})))
        session.add(Config(id="c2", type=ConfigType.TRAINING_ARGUMENTS, config=json.dumps({"k2": "v2"})))

    uuid4.return_value = "c3"

    response: AddConfigResponse = add_config(
        AddConfigRequest(
            type=ConfigType.TRAINING_ARGUMENTS,
            config='   \n {  "k3"   : "v3"\n\n}'
        ),
        dao=test_dao
    )
    assert response.config.id == "c3"
    assert response.config.config == json.dumps({"k3": "v3"})
    with test_dao.get_session() as session:
        assert len(session.query(Config).all()) == 3


def test_remove_config_happy():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(Config(id="c1", type=ConfigType.TRAINING_ARGUMENTS, config=json.dumps({"k1": "v1"})))
        session.add(Config(id="c2", type=ConfigType.TRAINING_ARGUMENTS, config=json.dumps({"k2": "v2"})))

    res = remove_config(RemoveConfigRequest(id="c1"), dao=test_dao)

    with test_dao.get_session() as session:
        assert len(session.query(Config).all()) == 1
