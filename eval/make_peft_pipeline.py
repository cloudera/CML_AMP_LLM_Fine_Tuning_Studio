import torch
from typing import Optional, Any
from transformers.utils import is_accelerate_available, is_bitsandbytes_available
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    pipeline
)
from peft import PeftModel


# TODO: pass in BitsAndBytes configuration as written in mlflow_pyfunc.py
def load_adapted_hf_generation_pipeline(
        base_model_name,
        lora_model_name,
        temperature: float = 0.7,
        top_p: float = 1.,
        max_tokens: int = 60,
        batch_size: int = 2,
        device: str = "cuda",
        load_in_8bit: bool = False,
        generation_kwargs: Optional[dict] = None,
):
    """
    Load a huggingface model & adapt with PEFT.
    Borrowed from https://github.com/tloen/alpaca-lora/blob/main/generate.py
    """

    if device == "cuda":
        if not is_accelerate_available():
            raise ValueError("Install `accelerate`")
    if load_in_8bit and not is_bitsandbytes_available():
        raise ValueError("Install `bitsandbytes`")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    task = "text-generation"

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_model_name,
            torch_dtype=torch.float16,
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

    if not load_in_8bit:
        model.half()
    model.eval()

    generation_kwargs = generation_kwargs if generation_kwargs is not None else {}
    config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_tokens,
        top_p=top_p,
        **generation_kwargs,
    )
    pipe = pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,  
        generation_config=config,
        framework="pt",
    )

    return pipe


def fetch_pipeline(model_name, adapter_name, device="gpu"):
    return load_adapted_hf_generation_pipeline(
        base_model_name=model_name,
        lora_model_name=adapter_name,
        device=device
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
