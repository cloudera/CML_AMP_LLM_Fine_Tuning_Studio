import mlflow
from transformers import GenerationConfig



class ModelLogger():

    def __init__(self) -> None:
        
        MLFLOW_TRACKING_URI="cml://localhost"   # this doesn't works somehow. Hence we need to start local mlflow server by running "mlflow server"
        mlflow.set_tracking_uri("http://localhost:5000") # when running local mlflow server
        mlflow.set_experiment("Evaluate MLFLOW")   # Name of MLFLOW experiment, this should be parameterized
        # Paramterize them
        self.config = GenerationConfig(
            do_sample=True,
            temperature=0.8,
            max_new_tokens=60,
            top_p=1
        )

        self.signature = mlflow.models.infer_signature(
            model_input="What are the three primary colors?",
            model_output="The three primary colors are red, yellow, and blue.",
        )

    def log_model(self, pipeline):
        with mlflow.start_run():
            model_info = mlflow.transformers.log_model(
                transformers_model=pipeline,
                torch_dtype='float16',
                artifact_path="custom-pipe",        # artifact_path can be dynamic
                signature=self.signature,
                registered_model_name="custom-pipe-chat",  # model_name can be dynamic
                model_config=self.config.to_dict()
            )
        return model_info