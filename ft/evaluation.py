from uuid import uuid4
import pathlib

import cmlapi
from cmlapi import CMLServiceApi
from ft.state import write_state, replace_state_field
from ft.api import *
from ft.consts import DEFAULT_FTS_GRPC_PORT

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

    arg_list.append("--base_model_id")
    arg_list.append(request.base_model_id)

    arg_list.append("--dataset_id")
    arg_list.append(request.dataset_id)

    arg_list.append("--base_model_id")
    arg_list.append(request.base_model_id)

    # Set Evaluation Dir argument
    arg_list.append("--result_dir")
    arg_list.append(result_dir)

    # Pass the IP address of the application engine that's running the FTS gRPC server.
    # passing this to the fine tuning job that's created allows the job to connect to
    # the gRPC server to request information about datasets, models, etc.
    arg_list.append("--fts_server_ip")
    arg_list.append(str(os.getenv("CDSW_IP_ADDRESS")))
    arg_list.append("--fts_server_port")
    arg_list.append(str(DEFAULT_FTS_GRPC_PORT))

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
    state = replace_state_field(state, evaluation_jobs=evaluation_jobs)
    return RemoveEvaluationJobResponse()
