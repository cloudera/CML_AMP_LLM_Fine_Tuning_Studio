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

    adapter_id: str
    """
    Adapter ID of the adapter that this job is training.
    """

    worker_props: Optional[FineTuningWorkerProps] = FineTuningWorkerProps()
    """
    Properties of each worker that will be spawned up.
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


class StartFineTuningJobResponse(BaseModel):
    job: Optional[FineTuningJobMetadata] = None

    


