from typing import Optional
from pydantic import BaseModel


class RegisteredModelMetadata(BaseModel):
    """
    Metadata about a registered model. This can
    apply right now to both adapters as well as
    models (TODO: need to check this logic)
    """

    cml_registered_model_id: Optional[str] = None
    """
    Model ID of the registered model.
    """

    mlflow_experiment_id: Optional[str] = None
    """
    MLFlow experiment ID. This allows us to extract individual
    model artifacts from the model registry, for example.
    """

    mlflow_run_id: Optional[str] = None
    """
    MLFlow run ID tied to this specific model artifact. This is used
    to extract individual model artifacts from MLFlow.
    """
