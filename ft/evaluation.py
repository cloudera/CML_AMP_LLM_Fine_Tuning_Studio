from uuid import uuid4
import pathlib
from ft.consts import BASE_MODEL_ONLY_ADAPTER_ID
import cmlapi
from cmlapi import CMLServiceApi
from ft.api import *
import os

from typing import List

from sqlalchemy import delete

from ft.db.dao import FineTuningStudioDao
from ft.db.model import EvaluationJob, Prompt, Config, Adapter, Model, Dataset


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


def _validate_start_evaluation_job_request(request: StartEvaluationJobRequest, dao: FineTuningStudioDao) -> None:
    # Check for required fields in StartEvaluationJobRequest
    required_fields = [
        "model_adapter_combinations", "dataset_id",
        "prompt_id", "adapter_bnb_config_id",
        "model_bnb_config_id", "generation_config_id",
        "cpu", "gpu", "memory"
    ]

    for field in required_fields:
        if not getattr(request, field):
            raise ValueError(f"Field '{field}' is required in StartEvaluationJobRequest.")

    # Ensure certain string fields are not empty after stripping out spaces
    string_fields = [
        "model_adapter_combinations", "dataset_id",
        "prompt_id", "adapter_bnb_config_id",
        "model_bnb_config_id", "generation_config_id"
    ]

    for field in string_fields:
        try:
            field_value = getattr(request, field).strip()
        except BaseException:
            # repeatable proto strip error
            field_value = getattr(request, field)
        if not field_value:
            raise ValueError(f"Field '{field}' cannot be an empty string or only spaces.")

    # Check if the referenced base_model_id exists in the database
    with dao.get_session() as session:

        # Check if the referenced dataset_id exists in the database
        if not session.query(Dataset).filter_by(id=request.dataset_id.strip()).first():
            raise ValueError(f"Dataset with ID '{request.dataset_id}' does not exist.")
        for model_adapter_combo in request.model_adapter_combinations:
            base_model_id = model_adapter_combo.base_model_id
            adapter_id = model_adapter_combo.adapter_id
            if not session.query(Model).filter_by(id=base_model_id.strip()).first():
                raise ValueError(f"Model with ID '{base_model_id}' does not exist.")
            # Check if the referenced adapter_id exists in the database or is the default base model
            if not session.query(Adapter).filter_by(id=adapter_id.strip()).first():
                if adapter_id == BASE_MODEL_ONLY_ADAPTER_ID:
                    pass
                else:
                    raise ValueError(f"Adapter with ID '{adapter_id}' does not exist.")

        # Check if the referenced prompt_id exists in the database
        if not session.query(Prompt).filter_by(id=request.prompt_id.strip()).first():
            raise ValueError(f"Prompt with ID '{request.prompt_id}' does not exist.")

        # Check if the referenced adapter_bnb_config_id exists in the database
        if not session.query(Config).filter_by(id=request.adapter_bnb_config_id.strip()).first():
            raise ValueError(f"Adapter BnB Config with ID '{request.adapter_bnb_config_id}' does not exist.")

        # Check if the referenced model_bnb_config_id exists in the database
        if not session.query(Config).filter_by(id=request.model_bnb_config_id.strip()).first():
            raise ValueError(f"Model BnB Config with ID '{request.model_bnb_config_id}' does not exist.")

        # Check if the referenced generation_config_id exists in the database
        if not session.query(Config).filter_by(id=request.generation_config_id.strip()).first():
            raise ValueError(f"Generation Config with ID '{request.generation_config_id}' does not exist.")


def get_comparison_adapter_id(model_adapter_combinations):
    """
    Given a list of ModelAdapterCombination, return the comparison adapter ID.
    """
    comparison_adapter_id = BASE_MODEL_ONLY_ADAPTER_ID
    for model_adapter_combo in model_adapter_combinations:
        if model_adapter_combo.adapter_id != BASE_MODEL_ONLY_ADAPTER_ID:
            comparison_adapter_id = model_adapter_combo.adapter_id
            break
    return comparison_adapter_id


def start_evaluation_job(request: StartEvaluationJobRequest,
                         cml: CMLServiceApi = None, dao: FineTuningStudioDao = None) -> StartEvaluationJobResponse:
    """
    Launch a CML Job which runs/orchestrates a finetuning operation.
    The CML Job itself does not run the finetuning work; it will launch a CML Worker(s) to allow
    more flexibility of parameters like CPU, memory, and GPU.
    """
    _validate_start_evaluation_job_request(request, dao)

    response = StartEvaluationJobResponse()

    # TODO: pull this and others into app state
    project_id = os.getenv("CDSW_PROJECT_ID")
    parent_job_id = str(uuid4())
    comparison_adapter_id = get_comparison_adapter_id(request.model_adapter_combinations)
    for idx, model_adapter_combo in enumerate(request.model_adapter_combinations):
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
        arg_list.append(model_adapter_combo.base_model_id)

        arg_list.append("--dataset_id")
        arg_list.append(request.dataset_id)

        arg_list.append("--base_model_id")
        arg_list.append(model_adapter_combo.base_model_id)

        arg_list.append("--adapter_id")
        arg_list.append(model_adapter_combo.adapter_id)

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

        arg_list.append("--selected_features")
        arg_list.append(request.selected_features)

        arg_list.append("--eval_dataset_fraction")
        arg_list.append(request.eval_dataset_fraction)

        arg_list.append("--comparison_adapter_id")
        arg_list.append(comparison_adapter_id)
        
        cpu = request.cpu
        gpu = request.gpu
        memory = request.memory
        gpu_label_id = request.gpu_label_id

        print(f"The args list of idx {idx} are \n{arg_list}\n\n")
        job_instance = cmlapi.models.create_job_request.CreateJobRequest(
            project_id=project_id,
            name=job_id,
            script=template_job.script,
            runtime_identifier=template_job.runtime_identifier,
            cpu=cpu,
            memory=memory,
            nvidia_gpu=gpu,
            arguments=" ".join([str(i).replace(" ", "") for i in arg_list])
        )

        # If provided, set accelerator label id for targeting gpu
        if gpu_label_id != -1:
            job_instance.accelerator_label_id = gpu_label_id

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
                base_model_id=model_adapter_combo.base_model_id,
                dataset_id=request.dataset_id,
                adapter_id=model_adapter_combo.adapter_id,
                prompt_id=request.prompt_id,
                num_workers=1,
                num_cpu=request.cpu,
                num_gpu=request.gpu,
                num_memory=request.memory,
                cml_job_id=created_job.id,
                parent_job_id=parent_job_id,
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
