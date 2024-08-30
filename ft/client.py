import grpc

from ft.api import *
from ft.proto.fine_tuning_studio_pb2_grpc import FineTuningStudioStub

from typing import List

import os


class FineTuningStudioClient:
    """
    Provides a name-friendly wrapper on the gRPC-generated stub.
    Smaller wrapper methods can be added here if there are repeated
    client patterns that appear in the code often (for example, the desire
    to grab the application state without constructing a request).

    Note that this class inherits from the stub, still. Which means you can
    access all standard stub requests as needed in "proper" request/response format.
    """

    def __init__(self, server_ip: str = None, server_port: str = None):
        if not server_ip:
            server_ip = os.getenv("FINE_TUNING_SERVICE_IP")
        if not server_port:
            server_port = os.getenv("FINE_TUNING_SERVICE_PORT")
        self.channel = grpc.insecure_channel(f"{server_ip}:{server_port}")
        self.stub = FineTuningStudioStub(self.channel)

        # Automatically wrap all gRPC methods with error handling
        for attr in dir(self.stub):
            if not attr.startswith('_') and callable(getattr(self.stub, attr)):
                setattr(self, attr, self._grpc_error_handler(getattr(self.stub, attr)))

    def _grpc_error_handler(self, func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except grpc.RpcError as error:
                raise ValueError(self._handle_grpc_error(error))
        return wrapper

    def _handle_grpc_error(self, error: grpc.RpcError):
        error_message = error.details()
        # Remove the "Exception calling application:" if it exists
        if error_message.startswith("Exception calling application:"):
            error_message = error_message.replace("Exception calling application:", "").strip()
        return error_message

    def get_datasets(self) -> List[DatasetMetadata]:
        return self.ListDatasets(ListDatasetsRequest()).datasets

    def get_prompts(self) -> List[PromptMetadata]:
        return self.ListPrompts(ListPromptsRequest()).prompts

    def get_models(self) -> List[ModelMetadata]:
        return self.ListModels(ListModelsRequest()).models

    def get_adapters(self) -> List[AdapterMetadata]:
        return self.ListAdapters(ListAdaptersRequest()).adapters

    def get_fine_tuning_jobs(self) -> List[FineTuningJobMetadata]:
        return self.ListFineTuningJobs(ListFineTuningJobsRequest()).fine_tuning_jobs

    def get_evaluation_jobs(self) -> List[EvaluationJobMetadata]:
        return self.ListEvaluationJobs(ListEvaluationJobsRequest()).evaluation_jobs
