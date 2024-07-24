from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel
import json 
from typing import Dict
from ft.state import get_state, update_state
from ft.prompt import PromptMetadata




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
    DatasetsManagerBase
)
from ft.state import AppState
from ft.dataset import ImportDatasetRequest, ImportDatasetResponse, DatasetType, DatasetMetadata
from ft.model import ImportModelRequest, ImportModelResponse, ModelMetadata
from ft.job import StartFineTuningJobRequest, StartFineTuningJobResponse, LocalFineTuningJobMetadata
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

    def __init__(self, 
                 datasets_manager: DatasetsManagerBase, 
                 models_manager: ModelsManagerBase, 
                 jobs_manager: FineTuningJobsManagerBase, 
                 state_location: str):
        self.state_location = state_location 
        self.datasets_manager = datasets_manager
        self.models_manager = models_manager
        self.jobs_manager = jobs_manager



class FineTuningApp():
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

    datasets: DatasetsManagerBase
    """
    Datasets adapter, for managing all of the datasets
    related to the dataset base.
    """

    def __init__(self, props: FineTuningAppProps):
        self.state_location = props.state_location
        self.models = props.models_manager
        self.jobs = props.jobs_manager
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
            jobs: List[LocalFineTuningJobMetadata] = state.jobs
            jobs.append(job_launch_response.job)
            update_state({"jobs": jobs})

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
