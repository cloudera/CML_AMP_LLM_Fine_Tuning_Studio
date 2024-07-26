from abc import ABC, abstractmethod
from uuid import uuid4
from typing import List

from ft.job import FineTuningJobMetadata, StartFineTuningJobRequest, StartFineTuningJobResponse
from ft.state import get_state, AppState, update_state
from ft.adapter import AdapterMetadata, AdapterType
import cmlapi
import os
import json
import pathlib


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
        job_id = str(uuid4())
        job_dir = ".app/job_runs/%s" % job_id

        pathlib.Path(job_dir).mkdir(parents=True, exist_ok=True)

        # Shortcut: lookup the template job created by the amp
        #  Use the template job to create any new jobs
        ft_base_job_id = self.cml_api_client.list_jobs(self.project_id,
                                                       search_filter='{"name":"Finetuning_Base_Job"}').jobs[0].id
        template_job = self.cml_api_client.get_job(
            project_id=self.project_id,
            job_id=ft_base_job_id
        )

        arg_list = []
        app_state = get_state()

        # Set Model argument
        # TODO: Support models that dont come from HF
        arg_list.append("--basemodel")
        hf_model = list(
            filter(
                lambda item: item.id == request.base_model_id,
                app_state.models))[0].huggingface_model_name
        arg_list.append(hf_model)

        # Set Dataset argument
        # TODO: Support datasets that dont come from HF
        arg_list.append("--dataset")
        hf_dataset = list(filter(lambda item: item.id == request.dataset_id, app_state.datasets))[0].huggingface_name
        arg_list.append(hf_dataset)

        # Set Prompt Text argument
        # TODO: Ideally this is just part of the aggregate config model below
        arg_list.append("--prompttemplate")
        prompt_text = list(filter(lambda item: item.id == request.prompt_id, app_state.prompts))[0].prompt_template
        with open("%s/%s" % (job_dir, "prompt.tmpl"), 'w') as prompt_text_file:
            prompt_text_file.write(prompt_text)
        arg_list.append("%s/%s" % (job_dir, "prompt.tmpl"))

        # Create aggregate config file containing all config object content
        # TODO: (lora, bnb, trainerargs, ?cmlworkersargs?)
        # Can we use pydantic here for serialization?
        print(request)
        aggregate_config = {
            "bnb_config": request.bits_and_bytes_config.to_dict()
        }
        print(aggregate_config)
        with open("%s/%s" % (job_dir, "job.config"), 'w') as aggregate_config_file:
            aggregate_config_file.write(json.dumps(aggregate_config, indent=4))
        arg_list.append("--aggregateconfig")
        arg_list.append("%s/%s" % (job_dir, "job.config"))

        arg_list.append("--experimentid")
        arg_list.append(job_id)

        out_dir = os.path.join(os.getenv("CUSTOM_LORA_ADAPTERS_DIR"), job_id)
        arg_list.append("--out_dir")
        arg_list.append(out_dir)

        # TODO: Support more args here: output-dir, bnb config, trainerconfig,
        # loraconfig, model, dataset, prompt_config, cpu, mem, gpu
        job_instance = cmlapi.models.create_job_request.CreateJobRequest(
            project_id=self.project_id,
            name=job_id,
            script=template_job.script,
            runtime_identifier=template_job.runtime_identifier,
            cpu=2,
            memory=8,
            nvidia_gpu=1,
            arguments=" ".join(arg_list)
        )
        print(job_instance)
        created_job = self.cml_api_client.create_job(
            body=job_instance,
            project_id=self.project_id
        )

        job_run = cmlapi.models.create_job_run_request.CreateJobRunRequest(
            project_id=self.project_id,
            job_id=created_job.id
        )

        launched_job = self.cml_api_client.create_job_run(
            body=job_run,
            project_id=self.project_id,
            job_id=created_job.id
        )

        metadata = FineTuningJobMetadata(
            out_dir=out_dir,
            start_time=launched_job.scheduling_at,
            job_id=job_id,
            base_model_id=request.base_model_id,
            dataset_id=request.dataset_id,
            prompt_id=request.prompt_id,
            num_workers=1,
            cml_job_id=created_job.id
        )

        if request.auto_add_adapter:
            adapter_metadata: AdapterMetadata = AdapterMetadata(
                id=str(uuid4()),
                name=request.adapter_name,
                type=AdapterType.LOCAL,
                model_id=request.base_model_id,
                location=out_dir,
                job_id=job_id,
                prompt_id=request.prompt_id,
            )

            state: AppState = get_state()
            adapters: List[AdapterMetadata] = state.adapters
            adapters.append(adapter_metadata)
            update_state({"adapters": adapters})

            metadata.adapter_id = adapter_metadata.id

        return StartFineTuningJobResponse(
            job=metadata
        )
