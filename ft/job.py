from pydantic import BaseModel
from ft.model import ModelMetadata
from typing import Optional
from transformers import BitsAndBytesConfig
from datetime import datetime
from enum import Enum


class FineTuningWorkerProps(BaseModel):
    num_cpu: int = 2
    num_memory: int = 8
    num_gpu: int = 1


class FineTuningJobStatus(Enum):
    SCHEDULED = "scheduled",
    RUNNING = "running",
    SUCCESS = "success",
    FAILURE = "failure"


class FineTuningJobMetadata(BaseModel):
    job_id: str
    """
    Unique job identifier of the job. For some job implementations (local
    fine tuning with the AMP), this job ID does not specifically have a
    CML counterpart or significance in the CDP ecosystem.
    """

    cml_job_id: str
    """
    CML identifier for the created CML job.
    """

    base_model_id: str
    """
    The model ID of the base model that should be used as a
    base for the fine tuning job.
    """

    dataset_id: str
    """
    The dataset that will be used to perform the training.
    This dataset ID is the App-specific ID.
    """

    prompt_id: str
    """
    The prompt that will be used for training. This is
    tied to the dataset for now, but that won't necessarily
    be a many-to-one relationship in the future.
    """

    num_workers: int
    """
    Number of workers to use for this fine-tuning job.
    """

    adapter_id: Optional[str] = None
    """
    Adapter ID of the adapter that this job is training.
    """

    worker_props: Optional[FineTuningWorkerProps] = FineTuningWorkerProps()
    """
    Properties of each worker that will be spawned up.
    """

    num_epochs: Optional[int] = 10
    """
    Number of epochs to run during fine-tuning.
    """

    learning_rate: Optional[float] = 2e-4
    """
    Learning rate to use during fine-tuning.
    """

class StartFineTuningJobRequest(BaseModel):

    adapter_name: str
    """
    Human-friendly identifier for the name of the adapter.
    """

    base_model_id: str
    """
    The model ID of the base model that should be used as a
    base for the fine tuning job.
    """

    dataset_id: str
    """
    The dataset that will be used to perform the training.
    This dataset ID is the App-specific ID.
    """

    prompt_id: str
    """
    The prompt that will be used for training. This is
    tied to the dataset for now, but that won't necessarily
    be a many-to-one relationship in the future.
    """

    num_workers: int
    """
    Number of workers to use for this fine-tuning job.
    """

    bits_and_bytes_config: Optional[BitsAndBytesConfig] = None
    """
    Bits and bytes config used to quantize the model. If this
    is present, then a model will be loaded with BnB config
    enabled.
    """

    auto_add_adapter: bool = True
    """
    Automatically add the trained job as an adapter to the app.
    """

    num_epochs: int
    """
    Number of epochs to run during fine-tuning.
    """

    learning_rate: float
    """
    Learning rate to use during fine-tuning.
    """

    cpu: int
    """
    Number of CPUs to allocate for this job.
    """

    gpu: int
    """
    Number of GPUs to allocate for this job.
    """

    memory: float
    """
    Amount of memory to allocate for this job (e.g., '16Gi').
    """

    train_test_split: Optional[float] = None
    """
    Optional dataset test split to split the dataset into a training
    dataset and an eval dataset. Evaluation datasets are used at epoch boundaries
    during training to compute metrics and compte loss again.
    """


class StartFineTuningJobResponse(BaseModel):
    job: Optional[FineTuningJobMetadata] = None
