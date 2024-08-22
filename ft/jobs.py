from uuid import uuid4

from ft.api import *
from ft.state import write_state, replace_state_field
import cmlapi
import os
import pathlib
from cmlapi import CMLServiceApi
import json


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
    arg_list.append("--base_model_id")
    arg_list.append(request.base_model_id)

    # Set Dataset argument
    arg_list.append("--dataset_id")
    arg_list.append(request.dataset_id)

    # Set Prompt Text argument
    # TODO: Ideally this is just part of the aggregate config model below
    arg_list.append("--prompt_id")
    arg_list.append(request.prompt_id)

    # Pass in all configs needed for FT join.
    # For now, ONLY THE BNB CONFIG ID of the ADAPTER will be used, however technically
    # we can have two different BnB configs for models vs adapters.
    arg_list.append("--bnb_config_id")
    arg_list.append(request.adapter_bnb_config_id)

    arg_list.append("--lora_config_id")
    arg_list.append(request.lora_config_id)

    arg_list.append("--training_arguments_config_id")
    arg_list.append(request.training_arguments_config_id)

    arg_list.append("--experimentid")
    arg_list.append(job_id)

    out_dir = os.path.join(request.output_dir, job_id)
    arg_list.append("--out_dir")
    arg_list.append(out_dir)

    arg_list.append("--train_out_dir")
    arg_list.append(os.path.join("outputs", job_id))

    arg_list.append("--adapter_name")
    arg_list.append(request.adapter_name)

    # Auto add the adapter to the database
    if request.auto_add_adapter:
        arg_list.append("--auto_add_adapter")

    if not request.train_test_split == StartFineTuningJobRequest().train_test_split:
        arg_list.append("--train_test_split")
        arg_list.append(str(request.train_test_split))

    if not request.dataset_fraction == StartFineTuningJobRequest().dataset_fraction:
        arg_list.append("--dataset_fraction")
        arg_list.append(str(request.dataset_fraction))

    hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
    if (not hf_token == "") and (hf_token is not None):
        arg_list.append("--hf_token")
        arg_list.append(hf_token)

    # If the user provides a custom user script, then change the actual script
    # that is running the fine tuning job.
    fine_tuning_script = template_job.script
    if not request.user_script == StartFineTuningJobRequest().user_script:
        fine_tuning_script = request.user_script

        # Determine the user configurations that are passed as part of this
        # custom user script. Serialized input configs are prioritized over
        # user config ids.
        user_config_id = request.user_config_id
        if not request.user_config == StartFineTuningJobRequest().user_config:
            user_config_md: ConfigMetadata = ConfigMetadata(
                id=str(uuid4()),
                type=ConfigType.CONFIG_TYPE_CUSTOM,
                config=json.dumps(json.loads(request.user_config))
            )
            state.configs.append(user_config_md)
            write_state(state)
            user_config_id = user_config_md.id

        # Pass the user config to the job.
        arg_list.append("--user_config_id")
        arg_list.append(user_config_id)

    cpu = request.cpu
    gpu = request.gpu
    memory = request.memory

    # TODO: Support more args here: output-dir, bnb config, trainerconfig,
    # loraconfig, model, dataset, prompt_config, cpu, mem, gpu
    job_instance = cmlapi.models.create_job_request.CreateJobRequest(
        project_id=project_id,
        name=job_id,
        script=fine_tuning_script,
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
        id=job_id,
        out_dir=request.output_dir,
        base_model_id=request.base_model_id,
        dataset_id=request.dataset_id,
        prompt_id=request.prompt_id,
        num_workers=request.num_workers,
        cml_job_id=created_job.id,
        num_epochs=request.num_epochs,
        learning_rate=request.learning_rate,
        dataset_fraction=request.dataset_fraction,
        train_test_split=request.train_test_split,
        num_cpu=request.cpu,
        num_gpu=request.gpu,
        num_memory=request.memory,
        training_arguments_config_id=request.training_arguments_config_id,
        lora_config_id=request.lora_config_id,
        model_bnb_config_id=request.model_bnb_config_id,
        adapter_bnb_config_id=request.adapter_bnb_config_id,
        user_script=request.user_script,
        user_config=request.user_config,
        user_config_id=request.user_config_id,
    )

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
    state = replace_state_field(state, fine_tuning_jobs=fine_tuning_jobs)
    return RemoveFineTuningJobResponse()
