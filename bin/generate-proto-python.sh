#!/bin/bash

# This script can be used to generate the protobuf
# definitions for python that can be used in the `ft`
# package. Note that the protobuf def is placed directly
# in the python package so we don't have to do any 
# manual package path manipulation. If we migrate this
# protobuf to be a first-class microservice, then this 
# workaround isn't necessary. 

# Generate the protobuf
python -m grpc_tools.protoc \
    -I=. \
    --python_out=. \
    --grpc_python_out=. \
    --pyi_out=. \
    ./ft/proto/fine_tuning_studio.proto


# Run autopep8 for formatting on the generated proto files.
# the generated proto files don't explicitly use autpep8 formatting,
# so we want to update this before we push changes. In reality, we
# should be running autopep8 formatting before every commit.

autopep8 ft/proto/fine_tuning_studio_pb2.py
autopep8 ft/proto/fine_tuning_studio_pb2_grpc.py
autopep8 ft/proto/fine_tuning_studio_pb2.pyi
