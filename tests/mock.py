from ft.managers.models import ModelsManagerBase
from ft.managers.datasets import DatasetsManagerBase
from ft.managers.jobs import FineTuningJobsManagerBase
from ft.managers.evaluation import MLflowEvaluationJobsManagerBase

from ft.app import *
from ft.mlflow import MLflowEvaluationJobMetadata, StartMLflowEvaluationJobRequest
from ft.model import *
from ft.job import *
from ft.adapter import *
from ft.prompt import *
from ft.dataset import *


class MockModelsManager(ModelsManagerBase):
    def export_model(self, request: ExportModelRequest) -> ExportModelResponse:
        pass

    def import_model(self, request: ImportModelRequest) -> ImportModelResponse:
        pass

    def list_models(self) -> List[ModelMetadata]:
        pass


class MockDatasetsManager(DatasetsManagerBase):
    def list_datasets(self) -> List[DatasetMetadata]:
        return super().list_datasets()

    def import_dataset(self, request: ImportDatasetRequest) -> ImportDatasetResponse:
        return super().import_dataset(request)

    def get_dataset(self, id: str) -> DatasetMetadata:
        return super().get_dataset(id)


class MockFineTuningJobsManager(FineTuningJobsManagerBase):
    def list_fine_tuning_jobs(self):
        return super().list_fine_tuning_jobs()

    def get_fine_tuning_job(self, job_id: str) -> FineTuningJobMetadata:
        return super().get_fine_tuning_job(job_id)

    def start_fine_tuning_job(self, request: StartFineTuningJobRequest):
        return super().start_fine_tuning_job(request)


class MockEvaluatorManager(MLflowEvaluationJobsManagerBase):
    def list_ml_flow_evaluation_jobs(self):
        return super().list_ml_flow_evaluation_jobs()

    def get_ml_flow_evaluation_job(self, job_id: str) -> MLflowEvaluationJobMetadata:
        return super().get_ml_flow_evaluation_job(job_id)

    def start_ml_flow_evaluation_job(self, request: StartMLflowEvaluationJobRequest):
        return super().start_ml_flow_evaluation_job(request)
