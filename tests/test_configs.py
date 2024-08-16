
import pytest

from ft.api import *

from unittest.mock import patch

from ft.configs import (
    get_config,
    list_configs,
    add_config,
    remove_config,
)

import json


def test_get_config_happy():

    test_configs = [
        ConfigMetadata(
            id="c1",
            type=ConfigType.CONFIG_TYPE_BITSANDBYTES_CONFIG,
            config=json.dumps({"load_in_8bit": True})
        ),
        ConfigMetadata(
            id="c2",
            type=ConfigType.CONFIG_TYPE_GENERATION_CONFIG,
            config=json.dumps({"do_sample": True})
        )
    ]
    test_state: AppState = AppState(
        configs=test_configs
    )

    out: ConfigMetadata = get_config(test_state, GetConfigRequest(id="c2")).config
    assert out.type == ConfigType.CONFIG_TYPE_GENERATION_CONFIG


def test_get_config_missing():

    test_configs = [
        ConfigMetadata(
            id="c1",
            type=ConfigType.CONFIG_TYPE_BITSANDBYTES_CONFIG,
            config=json.dumps({"load_in_8bit": True})
        ),
        ConfigMetadata(
            id="c2",
            type=ConfigType.CONFIG_TYPE_GENERATION_CONFIG,
            config=json.dumps({"do_sample": True})
        )
    ]
    test_state: AppState = AppState(
        configs=test_configs
    )

    with pytest.raises(AssertionError):
        out: ConfigMetadata = get_config(test_state, GetConfigRequest(id="c3")).config


def test_list_state_configs_happy():

    test_configs = [
        ConfigMetadata(
            id="c1",
            type=ConfigType.CONFIG_TYPE_BITSANDBYTES_CONFIG,
            config=json.dumps({"load_in_8bit": True})
        ),
        ConfigMetadata(
            id="c2",
            type=ConfigType.CONFIG_TYPE_GENERATION_CONFIG,
            config=json.dumps({"do_sample": True})
        )
    ]
    test_state: AppState = AppState(
        configs=test_configs
    )

    configs = list_configs(test_state, ListConfigsRequest()).configs
    assert len(configs) == 2
    assert configs[0] == test_configs[0]
    assert configs[1] == test_configs[1]


def test_list_state_configs_by_type():

    test_configs = [
        ConfigMetadata(
            id="c1",
            type=ConfigType.CONFIG_TYPE_BITSANDBYTES_CONFIG,
            config=json.dumps({"load_in_8bit": True})
        ),
        ConfigMetadata(
            id="c2",
            type=ConfigType.CONFIG_TYPE_GENERATION_CONFIG,
            config=json.dumps({"do_sample": True})
        )
    ]
    test_state: AppState = AppState(
        configs=test_configs
    )

    configs = list_configs(test_state, ListConfigsRequest(type=ConfigType.CONFIG_TYPE_GENERATION_CONFIG)).configs
    assert len(configs) == 1
    assert configs[0] == test_configs[1]


@patch("ft.configs.write_state")
@patch("ft.configs.uuid4")
def test_add_config_no_prior_configs(uuid4, write_state):
    test_state: AppState = AppState()

    test_config = json.dumps({"load_in_8bit": True})

    uuid4.return_value = "c1"

    response_config: ConfigMetadata = add_config(test_state, AddConfigRequest(
        type=ConfigType.CONFIG_TYPE_BITSANDBYTES_CONFIG,
        config=test_config
    )).config

    assert isinstance(response_config, ConfigMetadata)
    write_state.assert_called_with(AppState(
        configs=[
            ConfigMetadata(
                id="c1",
                type=ConfigType.CONFIG_TYPE_BITSANDBYTES_CONFIG,
                config='{"load_in_8bit": true}'
            )
        ]
    ))


@patch("ft.configs.write_state")
@patch("ft.configs.uuid4")
def test_add_config_existing_configs(uuid4, write_state):
    """
    Tests to make sure that two identical configs do not
    get stored into the config store. Note that formatting
    of the config string does not matter so long as the content is
    identical between the two configs.
    """

    test_config: ConfigMetadata = ConfigMetadata(
        id="c1",
        type=ConfigType.CONFIG_TYPE_GENERATION_CONFIG,
        config='{"max_new_tokens":1}'
    )

    test_state: AppState = AppState(
        configs=[
            test_config
        ]
    )

    response_config: AddConfigRequest = add_config(
        test_state,
        AddConfigRequest(
            type=ConfigType.CONFIG_TYPE_GENERATION_CONFIG,
            config='\n   { "max_new_tokens"    : 1\n\n\n}'
        )
    ).config

    write_state.assert_not_called()
    uuid4.assert_not_called()
    assert response_config == test_config


@patch("ft.configs.write_state")
@patch("ft.configs.uuid4")
def test_add_config_caching_mechanism_broke(uuid4, write_state):
    """
    Right now the config stores are designed as "cahcing" mechanism of
    sorts for basic configs (generation configs, quantization configs, etc.), which
    means that two identical configs should not exist in the datastore with different
    IDs. This might be allowed in the future (and is in fact allowed for other metadata
    types already like datasets, etc.), but given the caching type implementation of
    the add_config() feature for now, we'd like to mark this as an "antipattern"
    and discourage it (for now!).
    """

    test_configs = [
        ConfigMetadata(
            id="c1",
            type=ConfigType.CONFIG_TYPE_GENERATION_CONFIG,
            config=json.dumps({"load_in_8bit": True})
        ),
        ConfigMetadata(
            id="c2",
            type=ConfigType.CONFIG_TYPE_GENERATION_CONFIG,
            config=json.dumps({"load_in_8bit": True})
        )
    ]
    test_state: AppState = AppState(
        configs=test_configs
    )

    with pytest.raises(AssertionError):
        response: AddConfigResponse = add_config(
            test_state,
            AddConfigRequest(
                type=ConfigType.CONFIG_TYPE_GENERATION_CONFIG,
                config=json.dumps({"load_in_8bit": True})
            )
        )


@patch("ft.configs.write_state")
@patch("ft.configs.uuid4")
def test_add_config_existing_unique_configs(uuid4, write_state):
    test_configs = [
        ConfigMetadata(
            id="c1",
            type=ConfigType.CONFIG_TYPE_BITSANDBYTES_CONFIG,
            config=json.dumps({"load_in_8bit": True})
        ),
        ConfigMetadata(
            id="c2",
            type=ConfigType.CONFIG_TYPE_GENERATION_CONFIG,
            config=json.dumps({"do_sample": True})
        )
    ]
    test_state: AppState = AppState(
        configs=test_configs
    )

    uuid4.return_value = "c3"

    response: AddConfigResponse = add_config(
        test_state,
        AddConfigRequest(
            type=ConfigType.CONFIG_TYPE_GENERATION_CONFIG,
            config='   \n {  "top_k"   : 50\n\n}'
        )
    )

    write_state.assert_called_with(AppState(
        configs=[
            ConfigMetadata(
                id="c1",
                type=ConfigType.CONFIG_TYPE_BITSANDBYTES_CONFIG,
                config=json.dumps({"load_in_8bit": True})
            ),
            ConfigMetadata(
                id="c2",
                type=ConfigType.CONFIG_TYPE_GENERATION_CONFIG,
                config=json.dumps({"do_sample": True})
            ),
            ConfigMetadata(
                id="c3",
                type=ConfigType.CONFIG_TYPE_GENERATION_CONFIG,
                config='{"top_k": 50}'
            )
        ]
    ))


@patch("ft.configs.replace_state_field")
def test_remove_config_happy(replace_state_field):
    state: AppState = AppState(
        configs=[
            ConfigMetadata(
                id="g1"
            )
        ]
    )
    req = RemoveConfigRequest(id="g1")
    res = remove_config(state, req)
    replace_state_field.assert_called_with(state, configs=[])
