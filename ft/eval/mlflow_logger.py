import mlflow
from transformers import GenerationConfig
from mlflow.models import infer_signature
from ft.config.model_configs.config_loader import ModelMetadataFinder


class ModelLogger():

    def __init__(self, job_id) -> None:

        self.experiment_artifacts = mlflow.set_experiment(f"Run: {job_id}")
        # Paramterize them
        self.default_config = GenerationConfig(
            do_sample=True,
            temperature=0.8,
            max_new_tokens=60,
            top_p=1,
            bos_token_id=1
        )

        self.signature = self.set_schema_and_model_params_signature()

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

    def log_model_pipeline(self, pipeline, base_model_name, gen_config=None, adapter_name: str = None):
        if gen_config is None:
            gen_config = self.default_config
        else:
            if "bos_token_id" not in gen_config:
                gen_config["bos_token_id"] = ModelMetadataFinder.fetch_bos_token_id_from_config(base_model_name)
            if "eos_token_id" not in gen_config:
                gen_config["eos_token_id"] = ModelMetadataFinder.fetch_eos_token_id_from_config(base_model_name)
            # Deprecate the old max_length, which has an adverse affect on outputs
            # and truncation under the hood
            gen_config["max_length"] = None
            gen_config = GenerationConfig(**gen_config)
        full_name = f"{base_model_name}-{adapter_name}"
        full_name = full_name.replace("/","-")
        # truncate artifact path to last 49 charactere
        artifact_path = full_name[-49:]
        registered_model = "ft-model-" + str(artifact_path[-40:])

        with mlflow.start_run(run_name=full_name):
            model_info = mlflow.transformers.log_model(
                transformers_model=pipeline,
                torch_dtype='float16',
                artifact_path=artifact_path,        # artifact_path can be dynamic
                signature=self.signature,
                registered_model_name=registered_model,  # model_name can be dynamic
                model_config=gen_config.to_dict(),
                metadata={"model_full_name": full_name}
            )
        run_id =mlflow.search_runs(experiment_ids=[self.experiment_artifacts.experiment_id], filter_string=f"tags.mlflow.runName = '{full_name}'").run_id[0]
        return model_info, self.experiment_artifacts.experiment_id, run_id

    def log_model_multi_gpu(
            self,
            transformer_model,
            tokenizer_no_pad,
            gen_config=None,
            base_model_name: str = None,
            adapter_name: str = None):
        if gen_config is None:
            gen_config = self.default_config
        else:
            if "bos_token_id" not in gen_config:
                gen_config["bos_token_id"] = ModelMetadataFinder.fetch_bos_token_id_from_config(base_model_name)
            if "eos_token_id" not in gen_config:
                gen_config["eos_token_id"] = ModelMetadataFinder.fetch_eos_token_id_from_config(base_model_name)
            # Deprecate the old max_length, which has an adverse affect on outputs
            # and truncation under the hood
            gen_config["max_length"] = None
            gen_config = GenerationConfig(**gen_config)
        full_name = f"{base_model_name}-{adapter_name}"
        full_name = full_name.replace("/","-")
        # truncate artifact path to last 49 charactere
        artifact_path = full_name[-49:]
        registered_model = "ft-model-" + str(artifact_path[-40:])
        with mlflow.start_run(run_name=full_name):
            model_info = mlflow.transformers.log_model(
                transformers_model={"model": transformer_model, "tokenizer": tokenizer_no_pad},
                torch_dtype='float16',
                artifact_path=artifact_path,        # artifact_path can be dynamic
                signature=self.signature,
                registered_model_name=registered_model,  # model_name can be dynamic
                model_config=gen_config.to_dict(),
                metadata={"model_full_name": full_name}
            )
        run_id =mlflow.search_runs(experiment_ids=[self.experiment_artifacts.experiment_id], filter_string=f"tags.mlflow.runName = '{full_name}'").run_id[0]
        return model_info, self.experiment_artifacts.experiment_id, run_id


if __name__ == "__main__":
    c = ModelLogger()
