from abc import ABC, abstractmethod
from ft.job import FineTuningJobMetadata, StartFineTuningJobRequest

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
    
    def list_fine_tuning_jobs():
        pass

    def get_fine_tuning_job(job_id: str) -> FineTuningJobMetadata:
        return super().get_fine_tuning_job()
    
    def start_fine_tuning_job(request: StartFineTuningJobRequest):
        return super().start_fine_tuning_job()