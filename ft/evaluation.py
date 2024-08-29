from uuid import uuid4
import pathlib

import cmlapi
from cmlapi import CMLServiceApi
from ft.api import *

import os

from typing import List

from sqlalchemy import delete

from ft.db.dao import FineTuningStudioDao
from ft.db.model import EvaluationJob


def list_evaluation_jobs(request: ListEvaluationJobsRequest,
                         cml: CMLServiceApi = None, dao: FineTuningStudioDao = None) -> ListEvaluationJobsResponse:
    """
    In the future, this may handle filtering operations
    if we have a complex request object.
    """
    with dao.get_session() as session:
        jobs: List[EvaluationJob] = session.query(EvaluationJob).all()
        return ListEvaluationJobsResponse(
            evaluation_jobs=list(map(
                lambda x: x.to_protobuf(EvaluationJobMetadata),
                jobs
            ))
        )


def get_evaluation_job(request: GetEvaluationJobRequest,
                       cml: CMLServiceApi = None, dao: FineTuningStudioDao = None) -> GetEvaluationJobResponse:
    with dao.get_session() as session:
        return GetEvaluationJobResponse(
            evaluation_job=session
            .query(EvaluationJob)
            .where(EvaluationJob.id == request.id)
            .one()
            .to_protobuf(EvaluationJobMetadata)
        )


def start_evaluation_job(request: StartEvaluationJobRequest,
                         cml: CMLServiceApi = None, dao: FineTuningStudioDao = None) -> StartEvaluationJobResponse:
    """
    Launch a CML Job which runs/orchestrates a finetuning operation.
    The CML Job itself does not run the finetuning work; it will launch a CML Worker(s) to allow
    more flexibility of parameters like CPU, memory, and GPU.
    """

    response = StartEvaluationJobResponse()

    # TODO: pull this and others into app state
    project_id = os.getenv("CDSW_PROJECT_ID")

    job_id = str(uuid4())
    job_dir = f".app/mlflow_job_runs/{job_id}"

    result_dir = f"mlflow_results/{job_id}"

    pathlib.Path(job_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)

    # Lookup the template job created by the amp
    ft_base_job_id = cml.list_jobs(project_id,
                                   search_filter='{"name":"Mlflow_Evaluation_Base_Job"}').jobs[0].id
    template_job = cml.get_job(
        project_id=project_id,
        job_id=ft_base_job_id
    )

    arg_list = []

    arg_list.append("--base_model_id")
    arg_list.append(request.base_model_id)

    arg_list.append("--dataset_id")
    arg_list.append(request.dataset_id)

    arg_list.append("--base_model_id")
    arg_list.append(request.base_model_id)

    arg_list.append("--adapter_id")
    arg_list.append(request.adapter_id)

    arg_list.append("--prompt_id")
    arg_list.append(request.prompt_id)

    # Set Evaluation Dir argument
    arg_list.append("--result_dir")
    arg_list.append(result_dir)

    # Pass in configurations.
    arg_list.append("--adapter_bnb_config_id")
    arg_list.append(request.adapter_bnb_config_id)

    arg_list.append("--model_bnb_config_id")
    arg_list.append(request.model_bnb_config_id)

    arg_list.append("--generation_config_id")
    arg_list.append(request.generation_config_id)

    cpu = request.cpu
    gpu = request.gpu
    memory = request.memory

    print(arg_list)
    job_instance = cmlapi.models.create_job_request.CreateJobRequest(
        project_id=project_id,
        name=job_id,
        script=template_job.script,
        runtime_identifier=template_job.runtime_identifier,
        cpu=cpu,
        memory=memory,
        nvidia_gpu=gpu,
        arguments=" ".join(arg_list)
    )

    # Create job on CML
    created_job = cml.create_job(
        body=job_instance,
        project_id=project_id
    )

    # Launch job run
    job_run = cmlapi.models.create_job_run_request.CreateJobRunRequest(
        project_id=project_id,
        job_id=created_job.id
    )

    launched_job = cml.create_job_run(
        body=job_run,
        project_id=project_id,
        job_id=created_job.id
    )

    with dao.get_session() as session:
        eval_job: EvaluationJob = EvaluationJob(
            id=job_id,
            type=EvaluationJobType.MFLOW,
            base_model_id=request.base_model_id,
            dataset_id=request.dataset_id,
            adapter_id=request.adapter_id,
            prompt_id=request.prompt_id,
            num_workers=1,
            num_cpu=request.cpu,
            num_gpu=request.gpu,
            num_memory=request.memory,
            cml_job_id=created_job.id,
            evaluation_dir=result_dir,
            model_bnb_config_id=request.model_bnb_config_id,
            adapter_bnb_config_id=request.adapter_bnb_config_id,
            generation_config_id=request.generation_config_id,
        )
        session.add(eval_job)

        response = StartEvaluationJobResponse(
            evaluation_job=eval_job.to_protobuf(EvaluationJobMetadata)
        )

    return response


def remove_evaluation_job(request: RemoveEvaluationJobRequest,
                          cml: CMLServiceApi = None, dao: FineTuningStudioDao = None) -> RemoveEvaluationJobResponse:
    # TODO : Implement resource cleanup if required.
    with dao.get_session() as session:
        session.execute(delete(EvaluationJob).where(EvaluationJob.id == request.id))
    return RemoveEvaluationJobResponse()
