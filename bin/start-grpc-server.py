# start-grpc-server.py
from concurrent import futures
import logging
import grpc
from ft.proto import fine_tuning_studio_pb2_grpc
from ft.service import FineTuningStudioApp
from multiprocessing import Process
import socket 

def start_server(blocking: bool = False):
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fine_tuning_studio_pb2_grpc.add_FineTuningStudioServicer_to_server(FineTuningStudioApp(), server=server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    
    if blocking:
        server.wait_for_termination()

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        result = sock.connect_ex(('localhost', port))
        return result == 0
    
    
port = 50051
if not is_port_in_use(port):
    print("Starting up the gRPC server.")
    # Start the gRPC server if it's not already running
    start_server(blocking=True)
else:
    print("Server is already running.")