from concurrent import futures
import logging

import grpc
from ft.proto import fine_tuning_studio_pb2_grpc
from ft.server import FineTuningStudio


def start_server(blocking: bool = True):
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fine_tuning_studio_pb2_grpc.add_FineTuningStudioServicer_to_server(FineTuningStudio(), server=server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    
    if blocking:
        server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig()
    start_server()