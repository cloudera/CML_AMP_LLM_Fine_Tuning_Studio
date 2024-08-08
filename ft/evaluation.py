from uuid import uuid4
import pathlib

import cmlapi
from cmlapi import CMLServiceApi
from ft.state import write_state
from ft.api import *

import os


def list_evaluation_jobs(state: AppState, request: ListEvaluationJobsRequest,
                         cml: CMLServiceApi = None) -> ListEvaluationJobsResponse:
    """
    In the future, this may handle filtering operations
    if we have a complex request object.
    """
    return ListEvaluationJobsResponse(
        evaluation_jobs=state.evaluation_jobs
    )


def get_evaluation_job(state: AppState, request: GetEvaluationJobRequest,
                       cml: CMLServiceApi = None) -> GetEvaluationJobResponse:
    evaluation_jobs = list(filter(lambda x: x.id == request.id, state.evaluation_jobs))
    assert len(evaluation_jobs) == 1
    return GetEvaluationJobResponse(
        evaluation_job=evaluation_jobs[0]
    )


def start_evaluation_job(state: AppState, request: StartEvaluationJobRequest,
                         cml: CMLServiceApi = None) -> StartEvaluationJobResponse:
    """
    Launch a CML Job which runs/orchestrates a finetuning operation.
    The CML Job itself does not run the finetuning work; it will launch a CML Worker(s) to allow
    more flexibility of parameters like CPU, memory, and GPU.
    """

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

    # Set Model argument
    try:
        hf_model = next(item.huggingface_model_name for item in state.models if item.id == request.base_model_id)
    except StopIteration:
        raise ValueError(f"Base model with ID {request.base_model_id} not found.")
    arg_list.append("--basemodel")
    arg_list.append(hf_model)

    # Set Dataset argument
    try:
        hf_dataset = next(item.huggingface_name for item in state.datasets if item.id == request.dataset_id)
    except StopIteration:
        raise ValueError(f"Dataset with ID {request.dataset_id} not found.")
    arg_list.append("--dataset")
    arg_list.append(hf_dataset)

    # Set Adapter Path argument
    try:
        adapter_path = next(item.location for item in state.adapters if item.id == request.adapter_id)
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

    metadata = EvaluationJobMetadata(
        job_id=job_id,
        base_model_id=request.base_model_id,
        dataset_id=request.dataset_id,
        adapter_id=request.adapter_id,
        num_workers=1,
        worker_props=WorkerProps(
            num_cpu=request.cpu,
            num_gpu=request.gpu,
            num_memory=request.memory
        ),
        cml_job_id=created_job.id,
        evaluation_dir=result_dir
    )

    if not metadata == EvaluationJobMetadata():
        state.evaluation_jobs.append(metadata)
        write_state(state)

    return StartEvaluationJobResponse(job=metadata)


def remove_evaluation_job(state: AppState, request: RemoveEvaluationJobRequest,
                          cml: CMLServiceApi = None) -> RemoveEvaluationJobResponse:
    evaluation_jobs = list(filter(lambda x: not x.id == request.id, state.evaluation_jobs))
    write_state(AppState(
        datasets=state.datasets,
        prompts=state.prompts,
        adapters=state.adapters,
        fine_tuning_jobs=state.fine_tuning_jobs,
        evaluation_jobs=evaluation_jobs,
        models=state.models
    ))
    return RemoveEvaluationJobResponse()
