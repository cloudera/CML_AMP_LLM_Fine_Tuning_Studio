from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
import mlflow


class MLflowEvaluationWorkerProps(BaseModel):
    num_cpu: int = 2
    num_memory: int = 8
    num_gpu: int = 1


class MLflowEvaluationStatus(Enum):
    SCHEDULED = "scheduled"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"

class MLflowEvaluationJobMetadata(BaseModel):
    job_id: str
    """
    Unique job identifier for the evaluation job.
    """

    cml_job_id: str
    """
    CML identifier for the created CML job.
    """

    base_model_id: str
    """
    The model ID of the base model used for evaluation.
    """

    dataset_id: str
    """
    The dataset ID used for evaluation.
    """

    num_workers: int
    """
    Number of workers to use for this fine-tuning job.
    """

    adapter_id: Optional[str] = None
    """
    Adapter ID of the adapter that this job is training.
    """

    worker_props: Optional[MLflowEvaluationWorkerProps] = MLflowEvaluationWorkerProps()
    """
    Properties of each worker that will be spawned up.
    """

    start_time: datetime
    """
    The start time of the evaluation job.
    """

    end_time: Optional[datetime] = None
    """
    The end time of the evaluation job.
    """

    evaluation_dir: str
    """
    Path where the evaluation results will be stored.
    """


class StartMLflowEvaluationJobRequest(BaseModel):
    base_model_id: str
    """
    The model ID of the base model to be evaluated.
    """

    dataset_id: str
    """
    The dataset ID to be used for evaluation.
    """

    adapter_id: Optional[str] = None
    """
    Adapter ID of the adapter that this job is training.
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


class StartMLflowEvaluationJobResponse(BaseModel):
    job: Optional[MLflowEvaluationJobMetadata] = None
