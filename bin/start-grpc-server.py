# start-grpc-server.py
from concurrent import futures
import logging
import grpc
from ft.proto import fine_tuning_studio_pb2_grpc
from ft.service import FineTuningStudioApp
from ft.consts import DEFAULT_FTS_GRPC_PORT
from multiprocessing import Process
import socket 
import cmlapi
import os
import json
from typing import Dict

def start_server(blocking: bool = False):
    port = DEFAULT_FTS_GRPC_PORT
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fine_tuning_studio_pb2_grpc.add_FineTuningStudioServicer_to_server(FineTuningStudioApp(), server=server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    
    if blocking:
        server.wait_for_termination()



if __name__ == "__main__":
    
    # Make the fine tuning studio IP address and port available as project-level 
    # environment variables, so we can instantiate clients from anywhere 
    # within the project.
    
    cml = cmlapi.default_client()
    
    project_id = os.getenv("CDSW_PROJECT_ID")
    grpc_address = os.getenv("CDSW_IP_ADDRESS")
    grpc_port = DEFAULT_FTS_GRPC_PORT
    
    proj: cmlapi.Project = cml.get_project(project_id)
    proj_env: Dict = json.loads(proj.environment)
    proj_env.update({
        "FINE_TUNING_SERVICE_IP": grpc_address,
        "FINE_TUNING_SERVICE_PORT": grpc_port
    })
    
    updated_project: cmlapi.Project = cmlapi.Project(
        environment= json.dumps(proj_env)
    )
    out: cmlapi.Project = cml.update_project(updated_project, project_id=project_id)
    print(out.environment)

    # Start the server up. If this command fails (if the port is already
    # in use), the application script bin/start-app-script.sh will continue
    # to run and the error will exit gracefully.
    start_server(blocking=True)