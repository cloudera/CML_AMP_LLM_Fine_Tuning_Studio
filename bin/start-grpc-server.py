# start-grpc-server.py
from concurrent import futures
import logging
import grpc
from ft.proto import fine_tuning_studio_pb2_grpc
from ft.service import FineTuningStudioApp
from ft.consts import DEFAULT_FTS_GRPC_PORT
from multiprocessing import Process
import socket 

def start_server(blocking: bool = False):
    port = DEFAULT_FTS_GRPC_PORT
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fine_tuning_studio_pb2_grpc.add_FineTuningStudioServicer_to_server(FineTuningStudioApp(), server=server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    
    if blocking:
        server.wait_for_termination()


# Start the server up. If this command fails (if the port is already
# in use), the application script bin/start-app-script.sh will continue
# to run and the error will exit gracefully.
start_server(blocking=True)