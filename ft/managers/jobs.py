from abc import ABC, abstractmethod
from uuid import uuid4

from ft.job import FineTuningJobMetadata, StartFineTuningJobRequest, StartFineTuningJobResponse

import cmlapi
import os

class FineTuningJobsManagerBase(ABC):
    def __init__(self):
        return
    
    @abstractmethod
    def list_fine_tuning_jobs():
        pass 

    @abstractmethod
    def get_fine_tuning_job(job_id: str) -> FineTuningJobMetadata:
        pass

    @abstractmethod
    def start_fine_tuning_job(request: StartFineTuningJobRequest):
        pass
    

class FineTuningJobsManagerSimple(FineTuningJobsManagerBase):
    def __init__(self):
        # TODO: Maybe pull this out to a central startup process
        # Set up clients and base job ids
        # TODO: Also needs error handling/autrebuild of base job
        self.cml_api_client = cmlapi.default_client()
        self.project_id = os.getenv("CDSW_PROJECT_ID")
        self.ft_base_job_id = self.cml_api_client.list_jobs(self.project_id, 
                                      search_filter='{"name":"Finetuning_Base_Job"}').jobs[0].id
        
    
    def list_fine_tuning_jobs():
        pass

    def get_fine_tuning_job(job_id: str) -> FineTuningJobMetadata:
        return super().get_fine_tuning_job()
    

    def start_fine_tuning_job(self, request: StartFineTuningJobRequest):
        """
        Launch a CML Job which runs/orchestrates a finetuning operation
        The CML Job itself does not run the finetuning work, it will launch a CML Worker(s) to allow
        more flexibility of parameters like cpu,mem,gpu
        """
        new_job_id = str(uuid4())

        # Shortcut: lookup the template job created by the amp
        #  Use the template job to create any new jobs
        template_job = self.cml_api_client.get_job(
            project_id = self.project_id,
            job_id = self.ft_base_job_id
        )

        # TODO: Add more args here: output-dir, bnb config, trainerconfig, loraconfig, model, dataset, prompt_config, cpu, mem, gpu
        job_instance = cmlapi.models.create_job_request.CreateJobRequest(
            project_id = self.project_id,
            name = new_job_id,
            script = template_job.script,
            runtime_identifier = template_job.runtime_identifier,
            cpu = 2,
            memory = 8,
            nvidia_gpu = 1
        )
        print (job_instance)
        created_job = self.cml_api_client.create_job(
            body = job_instance,
            project_id = self.project_id
        )

        job_run = cmlapi.models.create_job_run_request.CreateJobRunRequest(
            project_id = self.project_id,
            job_id = created_job.id
        )

        launched_job = self.cml_api_client.create_job_run(
            body = job_run,
            project_id = self.project_id,
            job_id = self.ft_base_job_id
        )
        metadata = FineTuningJobMetadata(
            job_id = new_job_id,
            base_model_id = request.base_model_id,
            dataset_id = request.dataset_id,
            prompt_id = request.prompt_id,
            num_workers = 1,
            cml_job_id = created_job.id
        )

        return StartFineTuningJobResponse(
            job=metadata
        )
