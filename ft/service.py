import cmlapi


from ft.proto.fine_tuning_studio_pb2_grpc import FineTuningStudioServicer

from ft.db.dao import FineTuningStudioDao


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
    get_dataset_split_by_adapter
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
from ft.databse_ops import (
    export_database,
    import_database,
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

    def GetDatasetSplitByAdapter(self, request, context):
        return get_dataset_split_by_adapter(request, self.cml, self.dao)

    def ListModels(self, request, context):
        return list_models(request, self.cml, dao=self.dao)

    def GetModel(self, request, context):
        return get_model(request, self.cml, dao=self.dao)

    def AddModel(self, request, context):
        return add_model(request, self.cml, dao=self.dao)

    def ExportModel(self, request, context):
        return export_model(request, self.cml, dao=self.dao)

    def RemoveModel(self, request, context):
        return remove_model(request, self.cml, dao=self.dao)

    def ListAdapters(self, request, context):
        return list_adapters(request, self.cml, dao=self.dao)

    def GetAdapter(self, request, context):
        return get_adapter(request, self.cml, dao=self.dao)

    def AddAdapter(self, request, context):
        return add_adapter(request, self.cml, dao=self.dao)

    def RemoveAdapter(self, request, context):
        return remove_adapter(request, self.cml, dao=self.dao)

    def ListPrompts(self, request, context):
        return list_prompts(request, self.cml, dao=self.dao)

    def GetPrompt(self, request, context):
        return get_prompt(request, self.cml, dao=self.dao)

    def AddPrompt(self, request, context):
        return add_prompt(request, self.cml, dao=self.dao)

    def RemovePrompt(self, request, context):
        return remove_prompt(request, self.cml, dao=self.dao)

    def ListFineTuningJobs(self, request, context):
        return list_fine_tuning_jobs(request, self.cml, dao=self.dao)

    def GetFineTuningJob(self, request, context):
        return get_fine_tuning_job(request, self.cml, dao=self.dao)

    def StartFineTuningJob(self, request, context):
        return start_fine_tuning_job(request, self.cml, dao=self.dao)

    def RemoveFineTuningJob(self, request, context):
        return remove_fine_tuning_job(request, self.cml, dao=self.dao)

    def ListEvaluationJobs(self, request, context):
        return list_evaluation_jobs(request, self.cml, dao=self.dao)

    def GetEvaluationJob(self, request, context):
        return get_evaluation_job(request, self.cml, dao=self.dao)

    def StartEvaluationJob(self, request, context):
        return start_evaluation_job(request, self.cml, dao=self.dao)

    def RemoveEvaluationJob(self, request, context):
        return remove_evaluation_job(request, self.cml, dao=self.dao)

    def ListConfigs(self, request, context):
        return list_configs(request, dao=self.dao)

    def GetConfig(self, request, context):
        return get_config(request, dao=self.dao)

    def AddConfig(self, request, context):
        return add_config(request, dao=self.dao)

    def RemoveConfig(self, request, context):
        return remove_config(request, dao=self.dao)

    def ExportDatabase(self, request, context):
        return export_database(request, dao=self.dao)

    def ImportDatabase(self, request, context):
        return import_database(request, dao=self.dao)
