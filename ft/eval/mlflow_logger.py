import mlflow
from transformers import GenerationConfig
from mlflow.models import infer_signature
from uuid import uuid4


class ModelLogger():

    def __init__(self) -> None:

        # this doesn't works somehow. Hence we need to start local mlflow server by running "mlflow server"
        MLFLOW_TRACKING_URI = "cml://localhost"
        mlflow.set_tracking_uri("http://localhost:5000")  # when running local mlflow server
        # Name of MLFLOW experiment, this should be parameterized
        mlflow.set_experiment(f"Evaluate MLFLOW {str(uuid4())}")
        # Paramterize them
        self.default_config = GenerationConfig(
            do_sample=True,
            temperature=0.8,
            max_new_tokens=60,
            top_p=1,
            bos_token_id=1
        )

        self.signature = mlflow.models.infer_signature(
            model_input="What are the three primary colors?",
            model_output="The three primary colors are red, yellow, and blue.",
        )

    def set_schema_and_model_params_signature(self, input_example="", output_examples="", params=""):
        model_input = "What are the three primary colors?"
        model_output = "The three primary colors are red, yellow, and blue."
        signature = infer_signature(
            model_input=model_input,
            model_output=model_output,
            params={
                "max_new_tokens": 256,
                "repetition_penalty": 1.15,
                "return_full_text": False})
        return signature

    def log_model_pipeline(self, pipeline, gen_config=None):
        if gen_config is None:
            gen_config = self.default_config
        else:
            gen_config = GenerationConfig(**gen_config)
        with mlflow.start_run():
            model_info = mlflow.transformers.log_model(
                transformers_model=pipeline,
                torch_dtype='float16',
                artifact_path="custom-pipe",        # artifact_path can be dynamic
                signature=self.signature,
                registered_model_name="custom-pipe-chat",  # model_name can be dynamic
                model_config=gen_config.to_dict()
            )
        return model_info

    def log_model_multi_gpu(self, transformer_model, tokenizer_no_pad):
        # Log the model to MLflow
        # TODO: Check if this is needed or params from the infer_signature qorks correctly.
        config = GenerationConfig(
            do_sample=True,
            temperature=0.8,
            max_new_tokens=60,
            top_p=1
        )
        with mlflow.start_run():
            model_info = mlflow.transformers.log_model(
                transformers_model={"model": transformer_model, "tokenizer": tokenizer_no_pad},
                torch_dtype='float16',
                artifact_path="custom-pipe",        # artifact_path can be dynamic
                signature=self.set_schema_and_model_params_signature(),
                registered_model_name="custom-pipe-chat",
                model_config=self.config.to_dict()
            )

        return model_info


if __name__ == "__main__":
    c = ModelLogger()
