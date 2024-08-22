import cmlapi


from ft.proto.fine_tuning_studio_pb2_grpc import FineTuningStudioServicer

from ft.db.dao import FineTuningStudioDao

from ft.state import get_state
from ft.api import *

import os


from ft.datasets import (
    list_datasets,
    get_dataset,
    add_dataset,
    remove_dataset,
)

from ft.models import (
    list_models,
    get_model,
    add_model,
    export_model,
    remove_model,
)

from ft.adapters import (
    list_adapters,
    get_adapter,
    add_adapter,
    remove_adapter,
)

from ft.prompts import (
    list_prompts,
    get_prompt,
    add_prompt,
    remove_prompt,
)

from ft.jobs import (
    list_fine_tuning_jobs,
    get_fine_tuning_job,
    start_fine_tuning_job,
    remove_fine_tuning_job,
)

from ft.evaluation import (
    list_evaluation_jobs,
    get_evaluation_job,
    start_evaluation_job,
    remove_evaluation_job
)

from ft.configs import (
    list_configs,
    get_config,
    add_config,
    remove_config
)


class FineTuningStudioApp(FineTuningStudioServicer):
    """
    Top-Level gRPC Servicer for the Fine Tuning Studio app. This acts
    as an API surface to all gRPC interactions to the service. These
    methods primarily act as routers to application logic methods stored
    in other packages. When adding new API interactions, a new router
    is needed here that implements the gRPC Servicer base class. For development
    and testing simplicity, please try and keep application logic outside of
    this function, and assume that the state object passed to the method is
    non-mutable (in other words, application logic is responsible for writing
    state updates with write_state(...)).
    """

    def __init__(self):
        """
        Initialize the grpc server, and attach global connections
        (which include a CML client and a database DAO).
        """
        self.cml = cmlapi.default_client()

        self.dao = FineTuningStudioDao(engine_args={
            "pool_size": 5,
            "max_overflow": 10,
            "pool_timeout": 30,
            "pool_recycle": 1800,
        })

        # Load in environment variables
        self.project_id = os.getenv("CDSW_PROJECT_ID")
        self.engine_id = os.getenv("CDSW_ENGINE_ID")
        self.master_id = os.getenv("CDSW_MASTER_ID")
        self.master_ip = os.getenv("CDSW_MASTER_IP")
        self.domain = os.getenv("CDSW_DOMAIN")

    def ListDatasets(self, request, context):
        return list_datasets(request, self.cml, self.dao)

    def GetDataset(self, request, context):
        return get_dataset(request, self.cml, self.dao)

    def AddDataset(self, request, context):
        return add_dataset(request, self.cml, self.dao)

    def RemoveDataset(self, request, context):
        return remove_dataset(request, self.cml, self.dao)

    def ListModels(self, request, context):
        state: AppState = get_state()
        return list_models(state, request, self.cml)

    def GetModel(self, request, context):
        state: AppState = get_state()
        return get_model(state, request, self.cml)

    def AddModel(self, request, context):
        state: AppState = get_state()
        return add_model(state, request, self.cml)

    def ExportModel(self, request, context):
        state: AppState = get_state()
        return export_model(state, request, self.cml)

    def RemoveModel(self, request, context):
        state: AppState = get_state()
        return remove_model(state, request, self.cml)

    def ListAdapters(self, request, context):
        state: AppState = get_state()
        return list_adapters(state, request, self.cml)

    def GetAdapter(self, request, context):
        state: AppState = get_state()
        return get_adapter(state, request, self.cml)

    def AddAdapter(self, request, context):
        state: AppState = get_state()
        return add_adapter(state, request, self.cml)

    def RemoveAdapter(self, request, context):
        state: AppState = get_state()
        return remove_adapter(state, request, self.cml)

    def ListPrompts(self, request, context):
        state: AppState = get_state()
        return list_prompts(state, request, self.cml)

    def GetPrompt(self, request, context):
        state: AppState = get_state()
        return get_prompt(state, request, self.cml)

    def AddPrompt(self, request, context):
        state: AppState = get_state()
        return add_prompt(state, request, self.cml)

    def RemovePrompt(self, request, context):
        state: AppState = get_state()
        return remove_prompt(state, request, self.cml)

    def ListFineTuningJobs(self, request, context):
        return list_fine_tuning_jobs(request, self.cml, dao=self.dao)

    def GetFineTuningJob(self, request, context):
        return get_fine_tuning_job(request, self.cml, dao=self.dao)

    def StartFineTuningJob(self, request, context):
        return start_fine_tuning_job(request, self.cml, dao=self.dao)

    def RemoveFineTuningJob(self, request, context):
        return remove_fine_tuning_job(request, self.cml, dao=self.dao)

    def ListEvaluationJobs(self, request, context):
        state: AppState = get_state()
        return list_evaluation_jobs(state, request, self.cml)

    def GetEvaluationJob(self, request, context):
        state: AppState = get_state()
        return get_evaluation_job(state, request, self.cml)

    def StartEvaluationJob(self, request, context):
        state: AppState = get_state()
        return start_evaluation_job(state, request, self.cml)

    def RemoveEvaluationJob(self, request, context):
        state: AppState = get_state()
        return remove_evaluation_job(state, request, self.cml)

    def GetAppState(self, request, context):
        state: AppState = get_state()
        return GetAppStateResponse(
            state=state
        )

    def ListConfigs(self, request, context):
        return list_configs(request, dao=self.dao)

    def GetConfig(self, request, context):
        return get_config(request, dao=self.dao)

    def AddConfig(self, request, context):
        return add_config(request, dao=self.dao)

    def RemoveConfig(self, request, context):
        return remove_config(request, dao=self.dao)
