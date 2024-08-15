import json
from typing import Dict
from ft.api import AppState
import os
from google.protobuf.json_format import ParseDict, MessageToDict
from google.protobuf.message import Message
from google.protobuf.pyext._message import RepeatedCompositeContainer

# TODO: this should be an environment variable of the app.
DEFAULT_STATE_LOCATION = ".app/state.json"
"""
State location of the app. This contains all data that
is a project-specific session.
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
    state: AppState = ParseDict(state_data, AppState())
    return state


def write_state(state: AppState):
    """
    Write the app state to the state file.
    """

    state_data: Dict = MessageToDict(state, preserving_proto_field_name=True)
    with open(get_state_location(), "w") as f:
        f.write(json.dumps(state_data, indent=2))
    return


def replace_state_field(message: Message, **kwargs) -> Message:
    updated_message = type(message)()
    updated_message.MergeFrom(message)

    for field_name, new_value in kwargs.items():
        field = getattr(updated_message, field_name)
        if isinstance(field, (list, RepeatedCompositeContainer)):
            # Handle repeated fields
            del field[:]
            if isinstance(new_value, (list, RepeatedCompositeContainer)):
                field.extend(new_value)
            else:
                raise ValueError(f"The field '{field_name}' is repeated and expects a list.")
        else:
            # Handle single fields
            raise KeyError(
                f"We don't have a good way to do write single-field things like '{field_name}' to the app state.")
    write_state(updated_message)
    return updated_message
