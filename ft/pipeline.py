import torch
from transformers.utils import is_accelerate_available, is_bitsandbytes_available
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel
from typing import Dict


# TODO: pass in BitsAndBytes configuration as written in mlflow_pyfunc.py
def load_adapted_hf_generation_pipeline(
        base_model_name,
        lora_model_name,
        batch_size: int = 2,
        device: str = "cuda",
        bnb_config_dict: Dict = None,
        gen_config_dict: Dict = None,
):
    """
    Load a huggingface model & adapt with PEFT.
    Borrowed from https://github.com/tloen/alpaca-lora/blob/main/generate.py
    """

    if device == "cuda":
        if not is_accelerate_available():
            raise ValueError("Install `accelerate`")
    if bnb_config_dict and not is_bitsandbytes_available():
        raise ValueError("Install `bitsandbytes`")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    task = "text-generation"

    bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(**bnb_config_dict) if bnb_config_dict else None

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_model_name,
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_model_name,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, low_cpu_mem_usage=False
        )
        model = PeftModel.from_pretrained(
            model,
            lora_model_name
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()

    config = GenerationConfig(**gen_config_dict)
    pipe = pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        generation_config=config,
        framework="pt",
    )

    return pipe


def fetch_pipeline(model_name, adapter_name, device="gpu", bnb_config_dict: Dict = None, gen_config_dict: Dict = None):
    return load_adapted_hf_generation_pipeline(
        base_model_name=model_name,
        lora_model_name=adapter_name,
        device=device,
        bnb_config_dict=bnb_config_dict,
        gen_config_dict=gen_config_dict,
    )


if __name__ == "__main__":
    FOUNDATION_MODEL = "openlm-research/open_llama_3b"
    ADAPTER_NAME = "samwit/open-llama3B-4bit-lora"
    pipe = load_adapted_hf_generation_pipeline(
        base_model_name=FOUNDATION_MODEL,
        lora_model_name=ADAPTER_NAME,
        device="cpu"
    )
    print(pipe("Hello"))
