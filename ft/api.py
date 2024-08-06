"""
API Data types for the Fine Tuning Studio app.


The sole purpose of this file is to provide users with a cleanly-named
import for all of the datatypes created from the protobuf. Unfortunately,
protobuf does not allow for changing the name of generated python files,
and all files are appended with _pb2.py, which isn't clean. Using this
approach, the Streamlit frontend can access types generated from the
protobuf from "ft.api" instead!
"""


from ft.proto.fine_tuning_studio_pb2 import *
