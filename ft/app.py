from abc import ABC, abstractmethod
from typing import List
import json
from typing import Dict
from ft.state import get_state, write_state
import os

from ft.managers import (
    ModelsManagerBase,
    FineTuningJobsManagerBase,
    DatasetsManagerBase,
    MLflowEvaluationJobsManagerBase
)

from ft.api import *


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

    # TODO: Look into making this an API surface and eventually
    migrating to a protobuf
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
        # the app's dataset state with this data. The way we detect this in protobuf
        # is to compare the response message input to the default message, because in
        # protobuf3 there is no such concept as optional or None.
        if not import_response.dataset == DatasetMetadata():
            state: AppState = get_state()
            state.datasets.append(import_response.dataset)
            write_state(state)

        return import_response

    def remove_dataset(self, id: str):
        """
        TODO: this should be an official request/response type
        """
        state: AppState = get_state()
        datasets = list(filter(lambda x: not x.id == id, state.datasets))
        prompts = list(filter(lambda x: not x.dataset_id == id, state.prompts))
        write_state(AppState(
            datasets=datasets,
            prompts=prompts,
            adapters=state.adapters,
            jobs=state.jobs,
            mlflow=state.mlflow,
            models=state.models
        ))

    def add_prompt(self, prompt: PromptMetadata):
        """
        TODO: abstract this out similar to datasets above. Maybe have
        a prompt manager object part of the app.
        """
        state: AppState = get_state()
        state.prompts.append(prompt)
        write_state(state)

    def remove_prompt(self, id: str):
        state: AppState = get_state()
        prompts = list(filter(lambda x: not x.id == id, state.prompts))
        write_state(AppState(
            datasets=state.datasets,
            prompts=prompts,
            adapters=state.adapters,
            jobs=state.jobs,
            mlflow=state.mlflow,
            models=state.models
        ))

    def import_model(self, request: ImportModelRequest) -> ImportModelResponse:
        """
        Add a dataset to the App based on the request.
        """
        import_response: ImportModelResponse = self.models.import_model(request)

        # If we've successfully imported a new dataset, then make sure we update
        # the app's dataset state with this data. For now, using protobuf, we will
        # compare to the default value of the message of the internal model, which
        # means it was not set.
        if not import_response.model == ModelMetadata():
            state: AppState = get_state()
            state.models.append(import_response.model)
            write_state(state)

        return import_response

    def export_model(self, request: ExportModelRequest) -> ExportModelResponse:
        """
        Export a model and log metadata if appropriate
        """
        export_response: ExportModelResponse = self.models.export_model(request)

        # If we've successfully exported a model to model registry, add
        # this registered model to the app's metadata
        # TODO: adding this model to models list should be based on a setting
        # in the ExportModelRequest (similar to auto_add_adapter). This should NOT
        # just be based on model registry.
        if not export_response.model == ModelMetadata() and request.type == ModelType.MODEL_TYPE_MODEL_REGISTRY:
            state: AppState = get_state()
            state.models.append(export_response.model)
            write_state(state)

        return export_response

    def remove_model(self, id: str):
        """
        TODO: official request/response types
        """
        state: AppState = get_state()
        models = list(filter(lambda x: not x.id == id, state.models))
        write_state(AppState(
            datasets=state.datasets,
            prompts=state.prompts,
            adapters=state.adapters,
            jobs=state.jobs,
            mlflow=state.mlflow,
            models=models
        ))

    def launch_ft_job(self, request: StartFineTuningJobRequest) -> StartFineTuningJobResponse:
        """
        Create and launch a job for finetuning
        """
        job_launch_response: StartFineTuningJobResponse = self.jobs.start_fine_tuning_job(request)

        if not job_launch_response.job == StartFineTuningJobResponse().job:
            state: AppState = get_state()
            state.jobs.append(job_launch_response.job)
            write_state(state)

        return job_launch_response

    def launch_mlflow_job(self, request: StartMLflowEvaluationJobRequest) -> StartMLflowEvaluationJobResponse:
        """
        Create and launch a job for MLflow
        """
        job_launch_response: StartMLflowEvaluationJobResponse = self.mlflow.start_ml_flow_evaluation_job(request)

        if not job_launch_response.job == StartMLflowEvaluationJobResponse().job:
            state: AppState = get_state()
            state.mlflow.append(job_launch_response.job)
            write_state(state)

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
