from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel
import json
from typing import Dict
from ft.state import get_state, update_state
from ft.prompt import PromptMetadata
import os


'''
In order to get this UI demo working properly, we need two components:

1. a set of pydantic, data-driven schemas that define how data will be
passed around in the frontend. This is LIGHTWEIGHT and does NOT include
the scope of loading in actual models into memory on a machine. This is
a frontend component

2. A list of adapters that utilize the pydantic data structures that can
be used to initialize training jobs, view job statuses, etc. These adapters
can be implemented in any fashion (for example, we have mock adapters, which
then can be expanded to local training/inference adapters, which can then
be expanded out to CML-specific adapters (model registry, model serving)). These
adapters are responsible for example: listing models available, listing current
fine tuning jobs, etc.

'''

from ft.managers import (
    ModelsManagerBase,
    FineTuningJobsManagerBase,
    DatasetsManagerBase,
    MLflowEvaluationJobsManagerBase
)

from ft.state import AppState
from ft.dataset import ImportDatasetRequest, ImportDatasetResponse, DatasetType, DatasetMetadata
from ft.model import (
    ImportModelRequest,
    ImportModelResponse,
    ModelMetadata,
    ExportModelRequest,
    ExportModelResponse,
    ExportType,
    RegisteredModelMetadata
)
from ft.job import StartFineTuningJobRequest, StartFineTuningJobResponse, FineTuningJobMetadata
from ft.mlflow_job import StartMLflowEvaluationJobRequest, StartMLflowEvaluationJobResponse, MLflowEvaluationJobMetadata
from datasets import load_dataset


class FineTuningAppProps:
    state_location: str
    """
    Project-relative location of the app state. The app state contains a
    representation of the current state of an app and is intended to be used
    as a session store that can persist across browser sessions. This is GLOBAL
    to the project, so extra care is needed if multiple people are sharing
    the same project files (or if multiple workers are attempting to write
    to this file).
    """
    models_manager: ModelsManagerBase
    jobs_manager: FineTuningJobsManagerBase
    datasets_manager: DatasetsManagerBase
    mlflow_manager: MLflowEvaluationJobsManagerBase

    def __init__(self,
                 datasets_manager: DatasetsManagerBase,
                 models_manager: ModelsManagerBase,
                 jobs_manager: FineTuningJobsManagerBase,
                 mlflow_manager: MLflowEvaluationJobsManagerBase,
                 state_location: str = os.getenv("FINE_TUNING_APP_STATE_LOCATION")):
        self.state_location = state_location
        self.datasets_manager = datasets_manager
        self.models_manager = models_manager
        self.jobs_manager = jobs_manager
        self.mlflow_manager = mlflow_manager


class FineTuningApp():
    """
    This class acts as a backend API surface for the
    FT AMP application. Ideally all frontend requests should
    interact directly with methods found in this class, but occasionally
    lower-level operations can be performed directly by the variety
    of sub-managers for each component (models, datasets, etc.). By design,
    this app class also handles app state management, so only call sub-manager
    classes directly if you know what you're doing.
    """

    state_location: str
    """
    Location of the state file for the application.
    """

    models: ModelsManagerBase
    """
    Adapter for managing the models related to
    the Fine Tuning App.
    """

    jobs: FineTuningJobsManagerBase
    """
    Jobs manager for managing all ongoing fine
    tuning jobs in the FT app.
    """

    mlflow: MLflowEvaluationJobsManagerBase
    """
    Jobs manager for managing all ongoing fine
    mlflow jobs in the FT app.
    """

    datasets: DatasetsManagerBase
    """
    Datasets adapter, for managing all of the datasets
    related to the dataset base.
    """

    def __init__(self, props: FineTuningAppProps):
        self.state_location = props.state_location
        self.models = props.models_manager
        self.jobs = props.jobs_manager
        self.mlflow = props.mlflow_manager
        self.datasets = props.datasets_manager
        return

    def add_dataset(self, request: ImportDatasetRequest) -> ImportDatasetResponse:
        """
        Add a dataset to the App based on the request.
        """
        import_response: ImportDatasetResponse = self.datasets.import_dataset(request)

        # If we've successfully imported a new dataset, then make sure we update
        # the app's dataset state with this data.
        if import_response.dataset is not None:
            state: AppState = get_state()
            datasets: List[DatasetMetadata] = state.datasets
            datasets.append(import_response.dataset)
            update_state({"datasets": datasets})

        return import_response

    def remove_dataset(self, id: str):
        datasets = self.datasets.list_datasets()
        datasets = list(filter(lambda x: not x.id == id, datasets))
        update_state({"datasets": datasets})

        # Remove prompts related to this dataset
        prompts: List[PromptMetadata] = get_state().prompts
        prompts = prompts if prompts is not None else []
        prompts = list(filter(lambda x: not x.dataset_id == id, prompts))
        update_state({"prompts": prompts})

    def add_prompt(self, prompt: PromptMetadata):
        """
        TODO: abstract this out similar to datasets above. Maybe have
        a prompt manager object part of the app.
        """
        state: AppState = get_state()
        prompts: List[PromptMetadata] = state.prompts
        prompts.append(prompt)
        update_state({"prompts": prompts})

    def remove_prompt(self, id: str):
        prompts = get_state().prompts
        prompts = list(filter(lambda x: not x.id == id, prompts))
        update_state({"prompts": prompts})

    def import_model(self, request: ImportModelRequest) -> ImportDatasetResponse:
        """
        Add a dataset to the App based on the request.
        """
        import_response: ImportModelResponse = self.models.import_model(request)

        # If we've successfully imported a new dataset, then make sure we update
        # the app's dataset state with this data.
        if import_response.model is not None:
            state: AppState = get_state()
            lmodels: List[ModelMetadata] = state.models
            lmodels.append(import_response.model)
            update_state({"models": lmodels})

        return import_response

    def export_model(self, request: ExportModelRequest) -> ExportModelResponse:
        """
        Export a model and log metadata if appropriate
        """
        export_response: ExportModelResponse = self.models.export_model(request)

        # If we've successfully exported a model to model registry, add
        # this registered model to the app's metadata
        if export_response.registered_model is not None and request.type == ExportType.MODEL_REGISTRY:
            state: AppState = get_state()
            registered_models: List[RegisteredModelMetadata] = state.registered_models if state.registered_models is not None else [
            ]
            registered_models.append(export_response.registered_model)
            update_state({"registered_models": registered_models})

        return export_response

    def remove_model(self, id: str):
        """
        TODO: abstract this out to the model manager
        """
        models = self.models.list_models()
        models = list(filter(lambda x: not x.id == id, models))
        update_state({"models": models})

    def launch_ft_job(self, request: StartFineTuningJobRequest) -> StartFineTuningJobResponse:
        """
        Create and launch a job for finetuning
        """
        job_launch_response: StartFineTuningJobResponse = self.jobs.start_fine_tuning_job(request)

        if job_launch_response.job is not None:
            state: AppState = get_state()
            jobs: List[FineTuningJobMetadata] = state.jobs
            jobs.append(job_launch_response.job)
            update_state({"jobs": jobs})

        return job_launch_response

    def launch_mlflow_job(self, request: StartMLflowEvaluationJobRequest) -> StartMLflowEvaluationJobResponse:
        """
        Create and launch a job for MLflow
        """
        job_launch_response: StartMLflowEvaluationJobResponse = self.mlflow.start_ml_flow_evaluation_job(request)

        if job_launch_response.job is not None:
            state: AppState = get_state()
            jobs: List[MLflowEvaluationJobMetadata] = state.mlflow
            jobs.append(job_launch_response.job)
            update_state({"mlflow": jobs})

        return job_launch_response


INSTANCE: FineTuningApp = None
"""
This FT AMP follows a singleton pattern for the application logic wrapper
(includes all logic for loading datasets/models, starting/viewing FT jobs,
and evaluating models).

THIS IS NOT MEANT TO BE SET MANUALLY. Please use the create_app() method
to create the app ONCE per browser session, and subsequently use get_app()
to get a reference to the application.
"""


def create_app(props: FineTuningAppProps):
    """
    Wrapper for creating a fine-tuning app. This is called
    ONLY ONCE per project session and is defined at the entrypoint
    of the main app.
    """
    INSTANCE = FineTuningApp(props)
    return INSTANCE


def get_app():
    """
    Get the singleton instance of the app.
    """
    return INSTANCE
