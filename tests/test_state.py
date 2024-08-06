import os
import tempfile
from unittest.mock import patch
from google.protobuf.json_format import ParseDict
import json 

from ft.state import (
    get_state_location, 
    get_state, 
    write_state,
    DEFAULT_STATE_LOCATION
)
from ft.api import AppState, ModelType

class TestState():
    
    def test_get_state_location_no_env(self):
        if os.environ.get("FINE_TUNING_APP_STATE_LOCATION"):
            del os.environ["FINE_TUNING_APP_STATE_LOCATION"]
        state_location = get_state_location()
        assert state_location == DEFAULT_STATE_LOCATION
    
    def test_get_state_location_env(self):
        os.environ["FINE_TUNING_APP_STATE_LOCATION"] = "new/state/location.json"
        state_location = get_state_location()
        assert state_location == "new/state/location.json"
        
        
    def test_get_state_empty(self):
        os.environ["FINE_TUNING_APP_STATE_LOCATION"] = "tests/resources/test_state_empty.json"
        state: AppState = get_state()
        assert len(state.models) == 0
        assert len(state.adapters) == 0
        assert len(state.prompts) == 0
        
        
    def test_get_state_models(self):
        os.environ["FINE_TUNING_APP_STATE_LOCATION"] = "tests/resources/test_state_models.json"
        state: AppState = get_state()
        assert len(state.adapters) == 0
        assert len(state.models) == 1
        assert state.models[0].location == ""
        assert state.models[0].type == ModelType.MODEL_TYPE_HUGGINGFACE
        
    @patch("ft.state.get_state_location")
    def test_write_state(self, get_state_location):
        tmp = tempfile.NamedTemporaryFile()
        get_state_location.return_value = tmp.name 
        state: AppState = ParseDict({
            "models": [
                {
                    "id": "1fcb8f73-432b-4af7-abf0-1fae3e6579d5",
                    "type": 0,
                    "name": "Qwen/Qwen2-0.5B",
                    "huggingface_model_name": "Qwen/Qwen2-0.5B"
                }
            ]
        }, AppState())
        write_state(state)
        
        # Open the file
        json_in = json.load(open(tmp.name))
        assert json_in.get("models") is not None
        assert json_in.get("adapters") is None
        assert json_in["models"][0]["name"] == "Qwen/Qwen2-0.5B"
        