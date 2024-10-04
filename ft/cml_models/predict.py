from ft.eval.mlflow_pyfunc import MLFlowTransformers
import torch

mlt = MLFlowTransformers()


# singleton class to ensure that model is initialized only once
class SingletonModelFetcher:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.initialize_model()
        return cls._instance

    def initialize_model(self, base_model_name, peft_model_name):
        # Load model from disk or initialize it if necessary
        # This assumes that the model is already trained and saved in a file
        model, tokenizer = mlt.get_peft_model_and_tokenizer(base_model_name, peft_model_name)
        return model, tokenizer


modelFetcher = SingletonModelFetcher()
model, tokenizer = modelFetcher.initialize_model("huggingface/gpt2", "adapter/gpt2-ticketing")


def opt_args_value(args, arg_name, default):
    if arg_name in args.keys():
        return args[arg_name]
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
    temperature = float(opt_args_value(args, "temperature", 70))
    max_new_tokens = float(opt_args_value(args, "max_new_tokens", 50))
    top_p = float(opt_args_value(args, "top_p", 1.0))
    top_k = int(opt_args_value(args, "top_k", 0))
    repetition_penalty = float(opt_args_value(args, "repetition_penalty", 1.0))
    num_beams = int(opt_args_value(args, "num_beams", 1))

    return generate(prompt, max_new_tokens, temperature, repetition_penalty, num_beams, top_p, top_k)
