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
from ft.config.model_configs.config_loader import ModelMetadataFinder


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
    model_metadata_finder = ModelMetadataFinder(base_model_name)
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
        try:
            model = PeftModel.from_pretrained(
                model,
                lora_model_name,
                torch_dtype=torch.float16,
            )
        except ValueError:
            raise ValueError("Could not load Lora model due to invalid path or incompatibility")
        except TypeError as e:
            raise ValueError(f"Error loading Lora model due to error: {e}. Can load base model Only")
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        try:
            model = PeftModel.from_pretrained(
                model,
                lora_model_name,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        except ValueError:
            raise ValueError("Could not load Lora model due to invalid path or incompatibility")
        except TypeError as e:
            raise ValueError(f"Error loading Lora model due to error: {e}. Can load base model Only")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, low_cpu_mem_usage=False
        )
        try:
            model = PeftModel.from_pretrained(
                model,
                lora_model_name
            )
        except ValueError:
            raise ValueError("Could not load Lora model due to invalid path or incompatibility")
        except TypeError as e:
            raise ValueError(f"Error loading Lora model due to error: {e}. Can load base model Only")

    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = model_metadata_finder.fetch_bos_token_id_from_config(base_model_name)  # Todo: make this dynamic for different configs
    model.config.eos_token_id = 2

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


def fetch_pipeline(model_name, adapter_name, device="cuda", bnb_config_dict: Dict = None, gen_config_dict: Dict = None):
    return load_adapted_hf_generation_pipeline(
        base_model_name=model_name,
        lora_model_name=adapter_name,
        device=device,
        bnb_config_dict=bnb_config_dict,
        gen_config_dict=gen_config_dict,
    )


if __name__ == "__main__":
    FOUNDATION_MODEL = "bigscience/bloom-1b1"
    ADAPTER_NAME = "samwit/open-llama3B-4bit-lora"
    pipe = load_adapted_hf_generation_pipeline(
        base_model_name=FOUNDATION_MODEL,
        lora_model_name=ADAPTER_NAME,
        device="cuda",
        gen_config_dict={}
    )
    print(pipe("Hello"))
