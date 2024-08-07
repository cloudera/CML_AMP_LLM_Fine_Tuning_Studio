
import grpc

from ft.api import *
from ft.proto.fine_tuning_studio_pb2_grpc import FineTuningStudioStub


with grpc.insecure_channel('localhost:50051') as channel:
    stub = FineTuningStudioStub(channel=channel)
    datasets: ListDatasetsResponse = stub.ListDatasets(ListDatasetsRequest())
    print(datasets)
