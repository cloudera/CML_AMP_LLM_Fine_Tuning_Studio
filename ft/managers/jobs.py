from abc import ABC, abstractmethod
from uuid import uuid4

from ft.api import *
from ft.axolotl_config import AxolotlConfig
from ft.state import get_state, write_state
from ft.managers.cml import CMLManager
import cmlapi
import os
import json
import pathlib

from google.protobuf.json_format import MessageToDict


class FineTuningJobsManagerBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def list_fine_tuning_jobs(self):
        pass

    @abstractmethod
    def get_fine_tuning_job(self, job_id: str) -> FineTuningJobMetadata:
        pass

    @abstractmethod
    def start_fine_tuning_job(self, request: StartFineTuningJobRequest):
        pass


class FineTuningJobsManagerSimple(FineTuningJobsManagerBase, CMLManager):

    def __init__(self):
        CMLManager.__init__(self)

    def list_fine_tuning_jobs(self):
        pass

    def get_fine_tuning_job(self, job_id: str) -> FineTuningJobMetadata:
        return super().get_fine_tuning_job(job_id)


    def start_fine_tuning_job(self, request: StartFineTuningJobRequest):
        if request.finetuning_framework == 'legacy':
            return self.start_legacy_fine_tuning_job(request)
        elif request.finetuning_framework == 'axolotl':
            return self.start_axolotl_fine_tuning_job(request)
        else:
            raise ValueError("Unsupported finetuning framework: {}".format(request.finetuning_framework))

    def start_legacy_fine_tuning_job(self, request: StartFineTuningJobRequest):
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

        # TODO: see if the protobuf default value is sufficient here
        if not request.train_test_split == StartFineTuningJobRequest().train_test_split:
            arg_list.append("--train_test_split")
            arg_list.append(str(request.train_test_split))

        hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
        if (not hf_token == "") and (hf_token is not None):
            arg_list.append("--hf_token")
            arg_list.append(hf_token)

        arg_list.append("--finetuning_framework")
        arg_list.append(request.finetuning_framework)  # Convert to str

        cpu = request.cpu
        gpu = request.gpu
        memory = request.memory

        # TODO: Support more args here: output-dir, bnb config, trainerconfig,
        # loraconfig, model, dataset, prompt_config, cpu, mem, gpu
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
            ),
            finetuning_framework=request.finetuning_framework
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

            state: AppState = get_state()
            state.adapters.append(adapter_metadata)
            write_state(state)

            metadata.adapter_id = adapter_metadata.id

        return StartFineTuningJobResponse(
            FineTuningJob=metadata
        )

    def start_axolotl_fine_tuning_job(self, request: StartFineTuningJobRequest):
        """
        Launch a CML Job which runs/orchestrates a finetuning operation using the Axolotl framework.
        The CML Job itself does not run the finetuning work; it will launch a CML Worker(s) to allow
        more flexibility of parameters like cpu, mem, gpu.
        """
        axolotl_train_config = request.axolotl_train_config

        job_id = str(uuid4())
        job_dir = ".app/job_runs/{}".format(job_id)

        pathlib.Path(job_dir).mkdir(parents=True, exist_ok=True)

        # Lookup the template job created by the amp
        ft_base_job_id = self.cml_api_client.list_jobs(
            self.project_id, search_filter='{"name":"Finetuning_Base_Job"}'
        ).jobs[0].id
        template_job = self.cml_api_client.get_job(
            project_id=self.project_id, job_id=ft_base_job_id
        )

        app_state = get_state()

        # Set Model argument
        hf_model = next(
            (item.huggingface_model_name for item in app_state.models if item.id == request.base_model_id),
            None)
        if hf_model is None:
            raise ValueError("Base model ID not found in application state: {}".format(request.base_model_id))
        axolotl_train_config.base_model = hf_model

        # Set Dataset argument
        hf_dataset = next((item.huggingface_name for item in app_state.datasets if item.id == request.dataset_id), None)
        if hf_dataset is None:
            raise ValueError("Dataset ID not found in application state: {}".format(request.dataset_id))
        axolotl_train_config.datasets[0].path = hf_dataset

        # Set output directory
        out_dir = os.path.join(os.getenv("CUSTOM_LORA_ADAPTERS_DIR"), job_id)
        axolotl_train_config.output_dir = out_dir

        axolotl_train_config.mlflow_tracking_uri = "cml://localhost"
        axolotl_train_config.mlflow_experiment_name = job_id

        cpu = request.cpu
        gpu = request.gpu
        memory = request.memory

        axolotl_config = AxolotlConfig(axolotl_train_config)
        train_yaml_path = os.path.join(job_dir, "train.yaml")
        axolotl_config.save_to_yaml(train_yaml_path)

        arg_list = [
            "--experimentid", job_id,
            "--finetuning_framework", str(request.finetuning_framework),
            "--axolotl_yaml_file_path", train_yaml_path
        ]

        # Create and launch the CML job
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

        created_job = self.cml_api_client.create_job(
            body=job_instance, project_id=self.project_id
        )

        job_run = cmlapi.models.create_job_run_request.CreateJobRunRequest(
            project_id=self.project_id, job_id=created_job.id
        )

        launched_job = self.cml_api_client.create_job_run(
            body=job_run, project_id=self.project_id, job_id=created_job.id
        )

        metadata = FineTuningJobMetadata(
            out_dir=out_dir,
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
            ),
            finetuning_framework=request.finetuning_framework,
            axolotl_train_config=axolotl_config.get_config()
        )

        if request.auto_add_adapter:
            adapter_metadata = AdapterMetadata(
                id=str(uuid4()),
                name=request.adapter_name,
                type=AdapterType.ADAPTER_TYPE_PROJECT,
                model_id=request.base_model_id,
                location=out_dir,
                job_id=job_id,
                prompt_id=request.prompt_id,
            )
            state = get_state()
            state.adapters.append(adapter_metadata)
            write_state(state)
            metadata.adapter_id = adapter_metadata.id

        return StartFineTuningJobResponse(FineTuningJob=metadata)
