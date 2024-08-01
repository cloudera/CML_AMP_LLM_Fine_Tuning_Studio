from abc import ABC, abstractmethod
from uuid import uuid4
from typing import List
import os
import pathlib

import cmlapi
from ft.mlflow import (MLflowEvaluationJobMetadata, StartMLflowEvaluationJobRequest,
                           StartMLflowEvaluationJobResponse, MLflowEvaluationWorkerProps)
from ft.state import get_state, AppState
from ft.adapter import AdapterMetadata, AdapterType
from ft.managers.cml import CMLManager
from datetime import datetime


class MLflowEvaluationJobsManagerBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def list_ml_flow_evaluation_jobs(self):
        pass

    @abstractmethod
    def get_ml_flow_evaluation_job(self, job_id: str) -> MLflowEvaluationJobMetadata:
        pass

    @abstractmethod
    def start_ml_flow_evaluation_job(self, request: StartMLflowEvaluationJobRequest):
        pass


class MLflowEvaluationJobsManagerSimple(MLflowEvaluationJobsManagerBase, CMLManager):

    def list_ml_flow_evaluation_jobs(self):
        # Method to list ML flow evaluation jobs
        pass

    def get_ml_flow_evaluation_job(self, job_id: str) -> MLflowEvaluationJobMetadata:
        # Method to get a specific ML flow evaluation job
        return super().get_ml_flow_evaluation_job(job_id)

    def start_ml_flow_evaluation_job(self, request: StartMLflowEvaluationJobRequest):
        """
        Launch a CML Job which runs/orchestrates a finetuning operation.
        The CML Job itself does not run the finetuning work; it will launch a CML Worker(s) to allow
        more flexibility of parameters like CPU, memory, and GPU.
        """
        job_id = str(uuid4())
        job_dir = f".app/mlflow_job_runs/{job_id}"

        result_dir = f"mlflow_results/{job_id}"

        pathlib.Path(job_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)

        # Lookup the template job created by the amp
        ft_base_job_id = self.cml_api_client.list_jobs(self.project_id,
                                                       search_filter='{"name":"Mlflow_Evaluation_Base_Job"}').jobs[0].id
        template_job = self.cml_api_client.get_job(
            project_id=self.project_id,
            job_id=ft_base_job_id
        )

        arg_list = []
        app_state = get_state()

        # Set Model argument
        try:
            hf_model = next(item.huggingface_model_name for item in app_state.models if item.id == request.base_model_id)
        except StopIteration:
            raise ValueError(f"Base model with ID {request.base_model_id} not found.")
        arg_list.append("--basemodel")
        arg_list.append(hf_model)

        # Set Dataset argument
        try:
            hf_dataset = next(item.huggingface_name for item in app_state.datasets if item.id == request.dataset_id)
        except StopIteration:
            raise ValueError(f"Dataset with ID {request.dataset_id} not found.")
        arg_list.append("--dataset")
        arg_list.append(hf_dataset)

        # Set Adapter Path argument
        try:
            adapter_path = next(item.location for item in app_state.adapters if item.id == request.adapter_id)
        except StopIteration:
            raise ValueError(f"Adapter with ID {request.adapter_id} not found.")
        arg_list.append("--adapter_path")
        arg_list.append(adapter_path)

        # Set Evaluation Dir argument
        arg_list.append("--result_dir")
        arg_list.append(result_dir)

        cpu = request.cpu
        gpu = request.gpu
        memory = request.memory

        print(arg_list)
        job_instance = cmlapi.models.create_job_request.CreateJobRequest(
            project_id=self.project_id,
            name=job_id,
            script=template_job.script,
            runtime_identifier=template_job.runtime_identifier,
            cpu=cpu,
            memory=memory,
            nvidia_gpu=gpu,
            arguments=" ".join(arg_list)
        )

        # Create job on CML
        created_job = self.cml_api_client.create_job(
            body=job_instance,
            project_id=self.project_id
        )

        # Launch job run
        job_run = cmlapi.models.create_job_run_request.CreateJobRunRequest(
            project_id=self.project_id,
            job_id=created_job.id
        )

        launched_job = self.cml_api_client.create_job_run(
            body=job_run,
            project_id=self.project_id,
            job_id=created_job.id
        )

        metadata = MLflowEvaluationJobMetadata(
            start_time=launched_job.scheduling_at,
            job_id=job_id,
            base_model_id=request.base_model_id,
            dataset_id=request.dataset_id,
            adapter_id=request.adapter_id,
            num_workers=1,
            worker_props=MLflowEvaluationWorkerProps(
                cpu=request.cpu,
                memory=request.memory,
                gpu=request.gpu
            ),
            cml_job_id=created_job.id,
            evaluation_dir=result_dir
        )

        return StartMLflowEvaluationJobResponse(job=metadata)
