from pydantic import BaseModel
from typing import List
from ft.dataset import DatasetMetadata
from ft.job import FineTuningJobMetadata
from ft.model import ModelMetadata, RegisteredModelMetadata
from ft.prompt import PromptMetadata
from ft.adapter import AdapterMetadata
import json
from typing import Dict, Optional
import os


# TODO: this should be an environment variable of the app.
DEFAULT_STATE_LOCATION = ".app/state.json"
"""
State location of the app. This contains all data that
is a project-specific session.
"""


class AppState(BaseModel):
    """
    Basic app state class. This class
    can essentially be used as a project-wide session
    store for an application. Note that this is
    not explicitly a "browser" session, and stable behavior
    is not guaranteed when multiple users try to access
    the same project and the same session.

    To combat this, any call to FineTuningApp.get_state()
    will directly read again from the project state file.
    """

    datasets: Optional[List[DatasetMetadata]]
    """
    All available datasets associated with the application
    """

    models: Optional[List[ModelMetadata]]
    """
    All available models associated with the application
    """
    
    jobs: Optional[List[FineTuningJobMetadata]]
    """
    All available fine tuning jobs associated with the application
    """
    
    prompts: Optional[List[PromptMetadata]]
    """
    All available prompts associated with the application
    """
    
    adapters: Optional[List[AdapterMetadata]]
    """
    All available model adapters associated with the application
    """
    
    registered_models: Optional[List[RegisteredModelMetadata]]
    """
    All available CML registered models associated with the application
    """
    


def get_state_location():
    """
    Get the location of the currently loaded state file.
    """
    if os.environ.get("FINE_TUNING_APP_STATE_LOCATION"):
        return os.environ.get("FINE_TUNING_APP_STATE_LOCATION")
    return DEFAULT_STATE_LOCATION


def get_state():
    """
    Get the application's current state. This is a project-specific
    state, NOT a browser session specific state.

    This method gurantees to re-read the state of the app from the 
    state file every time this method is called.
    
    Currently, it's the responsibility of an App's adapters to update
    state data in the state file.
    """

    state_file = get_state_location()
    state_data = json.load(open(state_file))
    state = AppState(**state_data)
    return state


def write_state(state: AppState):
    """
    Write the app state to the state file.
    """

    state_data = state.model_dump_json(indent=2)
    with open(get_state_location(), "w") as f:
        f.write(state_data)
    return


def update_state(state_update: Dict):
    """
    Update the app's current state with any updated 
    state dict. This is used primarily in the App's wrapper
    functions that dispatch requests to adapters. This is to make 
    sure that state management is properly used.
    """

    state_json: Dict = get_state().model_dump()
    state_json.update(state_update)
    write_state(AppState(**state_json))