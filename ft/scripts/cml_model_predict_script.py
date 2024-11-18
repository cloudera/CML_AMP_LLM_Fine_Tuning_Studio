from ft.eval.mlflow_pyfunc import MLFlowTransformers
import torch
import os
import json

mlt = MLFlowTransformers()

# Main script used as the prediction/generation base of a deployed CML Model. This script
# globally loads a model and an adapter onto a CUDA-compatible GPU. The model exposes
# an "api_wrapper" function that expects a dictionary payload of an input prompt, as well
# as generation arguments.


# TODO: abstract out the BnB configuration that is loaded for this model. This should be an option
# that is available from the UI for each model that is deployed as a CML model.

# TODO: abstract out the generation args for a given model directly from the CML Model Export page
# in the UI. Still retain the option to modify these parameters as part of the inference request, but
# ensure the defaults are set by an environment variable (which is pulled from the request object).

# NOTE: This predict file currently uses the requirements file that exists in examples/examples-requirements.txt.
# any new required packages during inference should be added to that req file, not the base ./requirements.txt file.

# singleton class to ensure that model is initialized only once
class SingletonModelFetcher:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            # cls._instance.initialize_model() <--- just force a direct call to initialize_model() for now
        return cls._instance

    def initialize_model(self, base_model_name, peft_model_name):
        # Load model from disk or initialize it if necessary
        # This assumes that the model is already trained and saved in a file
        model, tokenizer = mlt.get_peft_model_and_tokenizer(base_model_name, peft_model_name)
        return model, tokenizer


print("fetching model and adapter parameters from environment...")
base_model_hf_name = os.getenv("FINE_TUNING_STUDIO_BASE_MODEL_HF_NAME")
adapter_location = os.getenv("FINE_TUNING_STUDIO_ADAPTER_LOCATION")
gen_config_dict = json.loads(os.getenv("FINE_TUNING_STUDIO_GEN_CONFIG_STRING"))

modelFetcher = SingletonModelFetcher()
model, tokenizer = modelFetcher.initialize_model(base_model_hf_name, adapter_location)


def opt_args_value(args, arg_name, default):
    if arg_name in args.keys():
        return args[arg_name]
    elif arg_name in gen_config_dict.keys():
        return gen_config_dict[arg_name]
    else:
        return default


def generate(prompt, max_new_tokens=128, temperature=0.7, repetition_penalty=1.0, num_beams=1, top_p=1.0, top_k=0):
    device = torch.device('cuda')
    input_tokens = tokenizer(prompt, return_tensors='pt').to(device)
    output_tokens = model.generate(**input_tokens,
                                   max_new_tokens=max_new_tokens,
                                   repetition_penalty=repetition_penalty,
                                   temperature=temperature,
                                   num_beams=num_beams,
                                   top_p=top_p,
                                   top_k=top_k)
    prompt_length = len(prompt)
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)[prompt_length:]


def api_wrapper(args):
    # Pick up args from model api
    prompt = args["prompt"]
    temperature = float(opt_args_value(args, "temperature", 0.5))
    max_new_tokens = float(opt_args_value(args, "max_new_tokens", 50))
    top_p = float(opt_args_value(args, "top_p", 1.0))
    top_k = int(opt_args_value(args, "top_k", 0))
    repetition_penalty = float(opt_args_value(args, "repetition_penalty", 1.0))
    num_beams = int(opt_args_value(args, "num_beams", 1))

    return generate(prompt, max_new_tokens, temperature, repetition_penalty, num_beams, top_p, top_k)


if __name__ == "__main__":
    print("Predict script initialized.")
