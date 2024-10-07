from uuid import uuid4
import cmlapi
from cmlapi import CMLServiceApi
from ft.api import *
import os


from ft.db.dao import FineTuningStudioDao
from ft.db.model import ExportJob, Adapter, Model


# def list_evaluation_jobs(request: ListEvaluationJobsRequest,
#                          cml: CMLServiceApi = None, dao: FineTuningStudioDao = None) -> ListEvaluationJobsResponse:
#     """
#     In the future, this may handle filtering operations
#     if we have a complex request object.
#     """
#     with dao.get_session() as session:
#         jobs: List[EvaluationJob] = session.query(EvaluationJob).all()
#         return ListEvaluationJobsResponse(
#             evaluation_jobs=list(map(
#                 lambda x: x.to_protobuf(EvaluationJobMetadata),
#                 jobs
#             ))
#         )


# def get_evaluation_job(request: GetEvaluationJobRequest,
#                        cml: CMLServiceApi = None, dao: FineTuningStudioDao = None) -> GetEvaluationJobResponse:
#     with dao.get_session() as session:
#         return GetEvaluationJobResponse(
#             evaluation_job=session
#             .query(EvaluationJob)
#             .where(EvaluationJob.id == request.id)
#             .one()
#             .to_protobuf(EvaluationJobMetadata)
#         )


def _validate_start_evaluation_job_request(request: ExportModelRequest, dao: FineTuningStudioDao) -> None:
    # Check for required fields in StartEvaluationJobRequest
    required_fields = [
        "type", "model_id", "adapter_id", "model_name"
    ]

    for field in required_fields:
        if not getattr(request, field):
            raise ValueError(f"Field '{field}' is required in StartEvaluationJobRequest.")

    # Ensure certain string fields are not empty after stripping out spaces
    string_fields = [
        "type", "model_id", "adapter_id", "model_name"
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

        base_model_id = request.model_id
        adapter_id = request.adapter_id
        if not session.query(Model).filter_by(id=base_model_id.strip()).first():
            raise ValueError(f"Model with ID '{base_model_id}' does not exist.")
        # Check if the referenced adapter_id exists in the database or is the default base model
        if not session.query(Adapter).filter_by(id=adapter_id.strip()).first():
            # if adapter_id == BASE_MODEL_ONLY_ADAPTER_ID:
            #     pass
            # else:
            # this is done because we shouldn't alow only base model exports.
            raise ValueError(f"Adapter with ID '{adapter_id}' does not exist.")

    # TODO: check if model_name exists in the database. This is necessary due to CML models constraints


def update_export_job_status(id, job_status, dao: FineTuningStudioDao = None):
    """
    Update the status of a export job in the database.
    """
    with dao.get_session() as session:
        job = session.query(ExportJob).filter_by(id=id).first()
        if not job:
            raise ValueError(f"Export job with ID '{id}' does not exist.")

        job.status = job_status
        session.commit()


def start_cml_export_job(request: ExportModelRequest,
                         cml: CMLServiceApi = None, dao: FineTuningStudioDao = None) -> ExportModelResponse:
    """
    Launch a CML Job which runs/orchestrates a finetuning operation.
    The CML Job itself does not run the finetuning work; it will launch a CML Worker(s) to allow
    more flexibility of parameters like CPU, memory, and GPU.
    """
    _validate_start_evaluation_job_request(request, dao)

    response = ExportModelResponse()

    # TODO: pull this and others into app state
    project_id = os.getenv("CDSW_PROJECT_ID")
    job_id = str(uuid4())

    # Lookup the template job created by the amp
    export_job_id = cml.list_jobs(project_id,
                                  search_filter='{"name":"CML_Export_Base_Job"}').jobs[0].id
    template_job = cml.get_job(
        project_id=project_id,
        job_id=export_job_id
    )

    arg_list = []

    arg_list.append("--id")
    arg_list.append(job_id)

    arg_list.append("--base_model_id")
    arg_list.append(request.base_model_id)

    arg_list.append("--adapter_id")
    arg_list.append(request.adapter_id)

    arg_list.append("--model_name")
    arg_list.append(request.model_name)

    arg_list.append("--model_description")
    arg_list.append(request.model_description)

    job_instance = cmlapi.models.create_job_request.CreateJobRequest(
        project_id=project_id,
        name=job_id,
        script=template_job.script,
        runtime_identifier=template_job.runtime_identifier,
        cpu=2,
        memory=8,
        nvidia_gpu=0,
        arguments=" ".join([str(i).replace(" ", "") for i in arg_list])
    )

    # # If provided, set accelerator label id for targeting gpu
    # if gpu_label_id != -1:
    #     job_instance.accelerator_label_id = gpu_label_id

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
        export_job: ExportJob = ExportJob(
            id=job_id,
            type=ModelExportType.CML_MODEL,
            base_model_id=request.base_model_id,
            adapter_id=request.adapter_id,
            cml_job_id=created_job.id,
            model_name=request.model_name,
        )
        session.add(export_job)

        response = ExportModelResponse(
            base_model_id=export_job.base_model_id,
            adapter_id=export_job.adapter_id
        )

    return response
