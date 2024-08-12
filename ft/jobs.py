from uuid import uuid4

from ft.api import *
from ft.state import write_state
from ft.consts import DEFAULT_FTS_GRPC_PORT
import cmlapi
import os
import json
import pathlib
from cmlapi import CMLServiceApi

from google.protobuf.json_format import MessageToDict


def list_fine_tuning_jobs(state: AppState, request: ListFineTuningJobsRequest,
                          cml: CMLServiceApi = None) -> ListFineTuningJobsResponse:
    """
    TODO: we can add filtering logic here.
    """
    return ListFineTuningJobsResponse(
        fine_tuning_jobs=state.fine_tuning_jobs
    )


def get_fine_tuning_job(state: AppState, request: GetFineTuningJobRequest,
                        cml: CMLServiceApi = None) -> GetFineTuningJobResponse:
    fine_tuning_jobs = list(filter(lambda x: x.id == request.id, state.fine_tuning_jobs))
    assert len(fine_tuning_jobs) == 1
    return GetFineTuningJobResponse(
        fine_tuning_job=fine_tuning_jobs[0]
    )


def start_fine_tuning_job(state: AppState, request: StartFineTuningJobRequest,
                          cml: CMLServiceApi = None) -> StartFineTuningJobResponse:
    """
    Launch a CML Job which runs/orchestrates a finetuning operation
    The CML Job itself does not run the finetuning work, it will launch a CML Worker(s) to allow
    more flexibility of parameters like cpu,mem,gpu
    """

    # TODO: pull this and others into app state
    project_id = os.getenv("CDSW_PROJECT_ID")

    job_id = str(uuid4())
    job_dir = ".app/job_runs/%s" % job_id

    pathlib.Path(job_dir).mkdir(parents=True, exist_ok=True)

    # Shortcut: lookup the template job created by the amp
    #  Use the template job to create any new jobs
    ft_base_job_id = cml.list_jobs(project_id,
                                   search_filter='{"name":"Finetuning_Base_Job"}').jobs[0].id
    template_job = cml.get_job(
        project_id=project_id,
        job_id=ft_base_job_id
    )

    arg_list = []

    # Set Model argument
    # TODO: Support models that dont come from HF
    arg_list.append("--basemodel")
    hf_model = list(
        filter(
            lambda item: item.id == request.base_model_id,
            state.models))[0].huggingface_model_name
    arg_list.append(hf_model)

    # Set Dataset argument
    arg_list.append("--dataset_id")
    arg_list.append(request.dataset_id)

    # Set Prompt Text argument
    # TODO: Ideally this is just part of the aggregate config model below
    arg_list.append("--prompttemplate")
    prompt_text = list(filter(lambda item: item.id == request.prompt_id, state.prompts))[0].prompt_template
    with open("%s/%s" % (job_dir, "prompt.tmpl"), 'w') as prompt_text_file:
        prompt_text_file.write(prompt_text)
    arg_list.append("%s/%s" % (job_dir, "prompt.tmpl"))

    # Create aggregate config file containing all config object content
    # TODO: (lora, bnb, trainerargs, ?cmlworkersargs?)
    # Right now we need to do a small conversion from BnB config to the protobuf message type.
    aggregate_config = {
        "bnb_config": MessageToDict(request.bits_and_bytes_config, preserving_proto_field_name=True)
    }
    with open("%s/%s" % (job_dir, "job.config"), 'w') as aggregate_config_file:
        aggregate_config_file.write(json.dumps(aggregate_config, indent=4))
    arg_list.append("--aggregateconfig")
    arg_list.append("%s/%s" % (job_dir, "job.config"))

    arg_list.append("--experimentid")
    arg_list.append(job_id)

    out_dir = os.path.join(os.getenv("CUSTOM_LORA_ADAPTERS_DIR"), job_id)
    arg_list.append("--out_dir")
    arg_list.append(out_dir)

    arg_list.append("--num_epochs")
    arg_list.append(str(request.num_epochs))  # Convert to str

    arg_list.append("--learning_rate")
    arg_list.append(str(request.learning_rate))  # Convert to str

    # Pass the IP address of the application engine that's running the FTS gRPC server.
    # passing this to the fine tuning job that's created allows the job to connect to
    # the gRPC server to request information about datasets, models, etc.
    arg_list.append("--fts_server_ip")
    arg_list.append(str(os.getenv("CDSW_IP_ADDRESS")))
    arg_list.append("--fts_server_port")
    arg_list.append(str(DEFAULT_FTS_GRPC_PORT))

    # TODO: see if the protobuf default value is sufficient here
    if not request.train_test_split == StartFineTuningJobRequest().train_test_split:
        arg_list.append("--train_test_split")
        arg_list.append(str(request.train_test_split))

    hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
    if (not hf_token == "") and (hf_token is not None):
        arg_list.append("--hf_token")
        arg_list.append(hf_token)

    cpu = request.cpu
    gpu = request.gpu
    memory = request.memory

    # TODO: Support more args here: output-dir, bnb config, trainerconfig,
    # loraconfig, model, dataset, prompt_config, cpu, mem, gpu
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

    created_job = cml.create_job(
        body=job_instance,
        project_id=project_id
    )

    job_run = cmlapi.models.create_job_run_request.CreateJobRunRequest(
        project_id=project_id,
        job_id=created_job.id
    )

    launched_job = cml.create_job_run(
        body=job_run,
        project_id=project_id,
        job_id=created_job.id
    )

    metadata = FineTuningJobMetadata(
        job_id=job_id,
        base_model_id=request.base_model_id,
        dataset_id=request.dataset_id,
        prompt_id=request.prompt_id,
        num_workers=request.num_workers,
        cml_job_id=created_job.id,
        num_epochs=request.num_epochs,
        learning_rate=request.learning_rate,
        worker_props=WorkerProps(
            num_cpu=request.cpu,
            num_gpu=request.gpu,
            num_memory=request.memory
        )
    )

    # TODO: ideally this should be done at the END of training
    # to ensure we're only loading WORKING adapters. Optionally,
    # we can track adapter training status in the metadata
    if request.auto_add_adapter:
        adapter_metadata: AdapterMetadata = AdapterMetadata(
            id=str(uuid4()),
            name=request.adapter_name,
            type=AdapterType.ADAPTER_TYPE_PROJECT,
            model_id=request.base_model_id,
            location=out_dir,
            job_id=job_id,
            prompt_id=request.prompt_id,
        )

        state.adapters.append(adapter_metadata)
        write_state(state)

        metadata.adapter_id = adapter_metadata.id

    # This check should always pass for now, but in the future,
    # we may consider handling "default" metadata message as a
    # fine tuning job creation failure, similar to how we do for
    # model and dataset application logic.
    if not metadata == FineTuningJobMetadata():
        state.fine_tuning_jobs.append(metadata)
        write_state(state)

    return StartFineTuningJobResponse(
        fine_tuning_job=metadata
    )


def remove_fine_tuning_job(state: AppState, request: RemoveFineTuningJobRequest,
                           cml: CMLServiceApi = None) -> RemoveFineTuningJobResponse:
    fine_tuning_jobs = list(filter(lambda x: not x.id == request.id, state.fine_tuning_jobs))
    write_state(AppState(
        datasets=state.datasets,
        prompts=state.prompts,
        adapters=state.adapters,
        fine_tuning_jobs=fine_tuning_jobs,
        evaluation_jobs=state.evaluation_jobs,
        models=state.models
    ))
    return RemoveFineTuningJobResponse()
