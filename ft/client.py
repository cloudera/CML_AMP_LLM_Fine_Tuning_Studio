
import grpc

from ft.api import *
from ft.proto.fine_tuning_studio_pb2_grpc import FineTuningStudioStub
from ft.consts import DEFAULT_FTS_GRPC_PORT

from typing import List


class FineTuningStudioClient(FineTuningStudioStub):
    """
    Provides a name-friendly wrapper on the gRPC-generated stub.
    Smaller wrapper methods can be added here if there are repeated
    client patterns that appear in the code often (for example, the desire
    to grab the application state without constructing a request).

    Note that this class inherits from the stub, still. Which means you can
    access all standard stub requests as needed in "proper" request/response format.
    """

    def __init__(self, server_ip: str = "localhost", server_port: str = DEFAULT_FTS_GRPC_PORT):
        self.channel = grpc.insecure_channel(f"{server_ip}:{server_port}")
        self.stub = FineTuningStudioStub.__init__(self, self.channel)

    def get_state(self) -> AppState:
        return self.GetAppState(GetAppStateRequest()).state

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
