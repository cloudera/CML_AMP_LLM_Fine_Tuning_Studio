from uuid import uuid4


from ft.api import *
import cmlapi
import os
import pathlib
from cmlapi import CMLServiceApi
import json

from ft.db.dao import FineTuningStudioDao
from ft.db.model import FineTuningJob, Config

from sqlalchemy import delete

from typing import List
from ft.db.model import Dataset, Prompt, Model

import yaml


def list_fine_tuning_jobs(request: ListFineTuningJobsRequest,
                          cml: CMLServiceApi = None, dao: FineTuningStudioDao = None) -> ListFineTuningJobsResponse:
    """
    TODO: we can add filtering logic here.
    """
    with dao.get_session() as session:
        ftjobs: List[FineTuningJob] = session.query(FineTuningJob).all()
        return ListFineTuningJobsResponse(
            fine_tuning_jobs=list(map(
                lambda x: x.to_protobuf(FineTuningJobMetadata),
                ftjobs
            ))
        )


def get_fine_tuning_job(request: GetFineTuningJobRequest,
                        cml: CMLServiceApi = None, dao: FineTuningStudioDao = None) -> GetFineTuningJobResponse:
    with dao.get_session() as session:
        return GetFineTuningJobResponse(
            fine_tuning_job=session
            .query(FineTuningJob)
            .where(FineTuningJob.id == request.id)
            .one()
            .to_protobuf(FineTuningJobMetadata)
        )


def _validate_fine_tuning_request(request: StartFineTuningJobRequest, dao: FineTuningStudioDao) -> None:
    """
    Validate the parameters of the StartFineTuningJobRequest.

    This function checks for necessary conditions and constraints in the request
    parameters and raises exceptions if validation fails.
    """

    # Validate framework_type
    if request.framework_type not in [FineTuningFrameworkType.LEGACY, FineTuningFrameworkType.AXOLOTL]:
        raise ValueError("Invalid framework type provided.")

    # Validate adapter_name
    if request.adapter_name and not request.adapter_name.replace("-", "").isalnum():
        raise ValueError("Adapter Name should only contain alphanumeric characters and hyphens, and no spaces.")

    # Validate output directory
    if request.output_dir and not os.path.isdir(request.output_dir):
        raise ValueError("Output Location must be a valid folder directory.")

    # Validate CPU, GPU, and memory allocations
    if request.cpu <= 0:
        raise ValueError("CPU allocation must be greater than 0.")
    if request.gpu < 0:
        raise ValueError("GPU allocation must be 0 or greater.")
    if request.memory <= 0:
        raise ValueError("Memory allocation must be greater than 0.")

    # Validate number of workers
    if request.num_workers <= 0:
        raise ValueError("Number of workers must be greater than 0.")

    # Validate number of epochs
    if request.num_epochs <= 0:
        raise ValueError("Number of epochs must be greater than 0.")

    # Validate learning rate
    if request.learning_rate <= 0:
        raise ValueError("Learning rate must be greater than 0.")

    # Validate dataset fraction (should be between 0 and 1)
    if not (0 < request.dataset_fraction <= 1):
        raise ValueError("Dataset fraction must be between 0 and 1.")

    # Validate train_test_split (should be between 0 and 1)
    if not (0 < request.train_test_split <= 1):
        raise ValueError("Train-test split must be between 0 and 1.")

    # Validate Axolotl config ID if using AXOLOTL framework
    if request.framework_type == FineTuningFrameworkType.AXOLOTL and not request.axolotl_config_id:
        raise ValueError("Axolotl Config ID is required for AXOLOTL framework type.")

    # Database validation for IDs
    with dao.get_session() as session:
        # Check if an adapter with the same name already exists in the database
        if request.adapter_name and session.query(FineTuningJob).filter_by(
                adapter_name=request.adapter_name.strip()).first():
            raise ValueError(f"An adapter with the name '{request.adapter_name}' already exists.")

        # Check if the referenced base_model_id exists in the database
        if not session.query(Model).filter_by(id=request.base_model_id.strip()).first():
            raise ValueError(f"Model with ID '{request.base_model_id}' does not exist.")

        # Check if the referenced dataset_id exists in the database
        if not session.query(Dataset).filter_by(id=request.dataset_id.strip()).first():
            raise ValueError(f"Dataset with ID '{request.dataset_id}' does not exist.")

        if request.framework_type == FineTuningFrameworkType.LEGACY:
            # Check if the referenced prompt_id exists in the database
            if request.prompt_id and not session.query(Prompt).filter_by(id=request.prompt_id.strip()).first():
                raise ValueError(f"Prompt with ID '{request.prompt_id}' does not exist.")

            # Check if the referenced training_arguments_config_id exists in the database
            if request.training_arguments_config_id and not session.query(
                    Config).filter_by(id=request.training_arguments_config_id.strip()).first():
                raise ValueError(
                    f"Training Arguments Config with ID '{request.training_arguments_config_id}' does not exist.")

            # Check if the referenced model_bnb_config_id exists in the database
            if request.model_bnb_config_id and not session.query(
                    Config).filter_by(id=request.model_bnb_config_id.strip()).first():
                raise ValueError(f"Model BnB Config with ID '{request.model_bnb_config_id}' does not exist.")

            # Check if the referenced adapter_bnb_config_id exists in the database
            if request.adapter_bnb_config_id and not session.query(
                    Config).filter_by(id=request.adapter_bnb_config_id.strip()).first():
                raise ValueError(f"Adapter BnB Config with ID '{request.adapter_bnb_config_id}' does not exist.")

            # Check if the referenced lora_config_id exists in the database
            if request.lora_config_id and not session.query(
                    Config).filter_by(id=request.lora_config_id.strip()).first():
                raise ValueError(f"Lora Config with ID '{request.lora_config_id}' does not exist.")

        if request.framework_type == FineTuningFrameworkType.AXOLOTL:
            # Check if the referenced axolotl_config_id exists in the database
            if request.axolotl_config_id and not session.query(
                    Config).filter_by(id=request.axolotl_config_id.strip()).first():
                raise ValueError(f"Axolotl Config with ID '{request.axolotl_config_id}' does not exist.")


def _build_argument_list(request: StartFineTuningJobRequest, job_id: str) -> List[str]:
    arg_list = [
        "--base_model_id", request.base_model_id,
        "--dataset_id", request.dataset_id,
        "--experimentid", job_id,
        "--out_dir", os.path.join(request.output_dir, job_id),
        "--train_out_dir", os.path.join("outputs", job_id),
        "--adapter_name", request.adapter_name,
        "--finetuning_framework_type", request.framework_type
    ]

    if request.prompt_id:
        arg_list.extend(["--prompt_id", request.prompt_id])
    if request.adapter_bnb_config_id:
        arg_list.extend(["--bnb_config_id", request.adapter_bnb_config_id])
    if request.lora_config_id:
        arg_list.extend(["--lora_config_id", request.lora_config_id])
    if request.training_arguments_config_id:
        arg_list.extend(["--training_arguments_config_id", request.training_arguments_config_id])
    if request.auto_add_adapter:
        arg_list.append("--auto_add_adapter")
    if request.train_test_split != StartFineTuningJobRequest().train_test_split:
        arg_list.extend(["--train_test_split", str(request.train_test_split)])
    if request.dataset_fraction != StartFineTuningJobRequest().dataset_fraction:
        arg_list.extend(["--dataset_fraction", str(request.dataset_fraction)])
    hf_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
    if hf_token:
        arg_list.extend(["--hf_token", hf_token])
    if request.axolotl_config_id:
        arg_list.extend(["--axolotl_config_id", request.axolotl_config_id])

    if request.framework_type == FineTuningFrameworkType.LEGACY:
        if request.num_workers:
            arg_list.extend(["--dist_num", request.num_workers])
        if request.cpu:
            arg_list.extend(["--dist_cpu", request.cpu])
        if request.memory:
            arg_list.extend(["--dist_mem", request.memory])
        if request.gpu:
            arg_list.extend(["--dist_gpu", request.gpu])
        if request.gpu_label_id:
            arg_list.extend(["--gpu_label_id", request.gpu_label_id])

    return arg_list


def _add_prompt_for_dataset(dataset_id: str, axolotl_config_id: str, dao: FineTuningStudioDao = None) -> str:
    if dao is None:
        raise ValueError("DAO object must be provided.")

    with dao.get_session() as session:
        # Fetch axolotl configuration
        axolotl_config = session.query(Config).filter(Config.id == axolotl_config_id).one_or_none()
        if not axolotl_config:
            raise ValueError(f"No configuration found with id {axolotl_config_id}.")

        try:
            axolotl_config_dict = yaml.safe_load(axolotl_config.config)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML configuration: {str(e)}")

        # Fetch dataset and dataset type
        dataset_type = axolotl_config_dict.get('datasets', [{}])[0].get('type', '')
        if not dataset_type:
            raise ValueError("Dataset type could not be found in the configuration.")

        dataset_type_config = session.query(Config).filter(Config.description == dataset_type).one_or_none()
        if not dataset_type_config:
            raise ValueError(f"No configuration found for dataset type {dataset_type}.")

        # Create the default template based on dataset features
        default_template = ""
        try:
            dataset_features = json.loads(dataset_type_config.config)
            for feature in dataset_features.keys():
                default_template += f"<{feature.capitalize()}>: {{{feature}}}\n"
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse dataset features: {str(e)}")

        dataset = session.query(Dataset).filter(Dataset.id == dataset_id).one_or_none()
        if not dataset:
            raise ValueError(f"No dataset found with id {dataset_id}.")

        # Check if the prompt already exists
        existing_prompt = session.query(Prompt).filter(
            Prompt.dataset_id == dataset_id,
            Prompt.prompt_template == default_template
        ).one_or_none()

        if existing_prompt:
            return existing_prompt.id

        # Create and add the new prompt
        try:
            prompt_id = str(uuid4())
            prompt = Prompt(
                id=prompt_id,
                name=f"AXOLOTL_AUTOGENERATED : {dataset_type}_{dataset.name}",
                dataset_id=dataset_id,
                prompt_template=default_template
            )
            session.add(prompt)
            session.commit()
            return prompt_id
        except Exception as e:
            raise ValueError(f"Error occurred while adding prompt: {str(e)}")


def start_fine_tuning_job(request: StartFineTuningJobRequest,
                          cml: CMLServiceApi = None, dao: FineTuningStudioDao = None) -> StartFineTuningJobResponse:
    """
    Launch a CML Job which runs/orchestrates a finetuning operation
    The CML Job itself does not run the finetuning work, it will launch a CML Worker(s) to allow
    more flexibility of parameters like cpu,mem,gpu
    """

    # Specify model framework type.
    if 'framework_type' not in [x[0].name for x in request.ListFields()]:
        framework_type = FineTuningFrameworkType.LEGACY
    else:
        framework_type: FineTuningFrameworkType = request.framework_type

    request.framework_type = framework_type

    # Validate the request parameters
    _validate_fine_tuning_request(request, dao)

    # TODO: pull this and others into app state
    project_id = os.getenv("CDSW_PROJECT_ID")

    job_id = str(uuid4())
    job_dir = ".app/job_runs/%s" % job_id

    if framework_type == FineTuningFrameworkType.LEGACY:
        base_job_name = "Accel_Finetuning_Base_Job"
    else:
        base_job_name = "Finetuning_Base_Job"

    pathlib.Path(job_dir).mkdir(parents=True, exist_ok=True)

    # Shortcut: lookup the template job created by the amp
    #  Use the template job to create any new jobs
    ft_base_job_id = cml.list_jobs(project_id,
                                   search_filter='{"name":"%s"}' % base_job_name).jobs[0].id
    template_job = cml.get_job(
        project_id=project_id,
        job_id=ft_base_job_id
    )

    if request.framework_type == FineTuningFrameworkType.AXOLOTL:
        try:
            prompt_id = _add_prompt_for_dataset(request.dataset_id, request.axolotl_config_id, dao)
            if prompt_id:  # Check if prompt_id is not None or empty
                request.prompt_id = prompt_id
            else:
                # Handle the case where prompt_id is None or an empty string
                raise ValueError("Prompt ID could not be generated or found.")
        except Exception as e:
            # Handle exceptions, log the error, and set a fallback or take appropriate action
            print(f"Failed to add prompt for dataset: {str(e)}")
            # Optionally set request.prompt_id to a default value or take another action
            request.prompt_id = None

    arg_list = _build_argument_list(request, job_id)

    # If the user provides a custom user script, then change the actual script
    # that is running the fine tuning job.
    fine_tuning_script = template_job.script
    user_config_id = request.user_config_id
    if not request.user_script == StartFineTuningJobRequest().user_script:
        fine_tuning_script = request.user_script

        # Determine the user configurations that are passed as part of this
        # custom user script. Serialized input configs are prioritized over
        # user config ids.
        user_config_id = request.user_config_id
        if not request.user_config == StartFineTuningJobRequest().user_config:
            user_config: Config = Config(
                id=str(uuid4()),
                type=ConfigType.CUSTOM,
                config=json.dumps(json.loads(request.user_config)),
            )
            with dao.get_session() as session:
                session.add(user_config)
                user_config_id = user_config.id

        # Pass the user config to the job.
        arg_list.append("--user_config_id")
        arg_list.append(user_config_id)

    cpu = request.cpu
    gpu = request.gpu
    memory = request.memory
    gpu_label_id = request.gpu_label_id

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
        arguments=" ".join([str(i).replace(" ", "") for i in arg_list])
    )

    # If provided, set accelerator label id for targeting gpu
    if gpu_label_id != -1:
        job_instance.accelerator_label_id = gpu_label_id

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

    ftjob: FineTuningJob = FineTuningJob(
        id=job_id,
        framework_type=framework_type,
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
        user_config_id=user_config_id,
        axolotl_config_id=request.axolotl_config_id,
        gpu_label_id=request.gpu_label_id,
        adapter_name=request.adapter_name
    )

    response = StartFineTuningJobResponse()
    with dao.get_session() as session:
        session.add(ftjob)
        response = StartFineTuningJobResponse(
            fine_tuning_job=ftjob.to_protobuf(FineTuningJobMetadata)
        )

    return response


def remove_fine_tuning_job(request: RemoveFineTuningJobRequest,
                           cml: CMLServiceApi = None, dao: FineTuningStudioDao = None) -> RemoveFineTuningJobResponse:
    # TODO : To cleanup the job runs folders and CML workspace jobs.
    with dao.get_session() as session:
        session.execute(delete(FineTuningJob).where(FineTuningJob.id == request.id))
    return RemoveFineTuningJobResponse()
