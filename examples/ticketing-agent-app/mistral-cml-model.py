import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os
import torch
import cml.models_v1 as models

BASE_MODEL = "unsloth/mistral-7b-instruct-v0.3"
DEVICE = torch.device('cuda')

bnb_config = BitsAndBytesConfig(
                # Load the model with 4-bit quantization
                load_in_4bit=True,
                # Use double quantization
                bnb_4bit_use_double_quant=True,
                # Use 4-bit Normal Float for storing the base model weights in GPU memory
                bnb_4bit_quant_type="nf4",
                # De-quantize the weights to 16-bit (Brain) float before the forward/backward pass
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

# Model + Tokenizer Setup
PREBUILT_LORA_ADAPTER_DIR = "/home/cdsw/examples/ticketing-agent-app/adapter/mistral-7b-ticketing"
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL,
                                            return_dict=True,
                                            quantization_config=bnb_config)

model = PeftModel.from_pretrained(model=model,
                                model_id=PREBUILT_LORA_ADAPTER_DIR,
                                adapter_name=PREBUILT_LORA_ADAPTER_DIR)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def opt_args_value(args, arg_name, default):
  if arg_name in args.keys():
    return args[arg_name]
  else:
    return default

def generate(prompt, max_new_tokens=128, temperature=0.7, repetition_penalty=1.0, num_beams=1, top_p=1.0, top_k=0):
    device = torch.device('cuda')
    input_tokens = tokenizer(prompt, return_tensors='pt').to(DEVICE)
    output_tokens = model.generate(**input_tokens,
                                    max_new_tokens=max_new_tokens,
                                    repetition_penalty=repetition_penalty,
                                    temperature=temperature,
                                    num_beams=num_beams,
                                    top_p=top_p,
                                    top_k=top_k)
    prompt_length = len(prompt)
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)[prompt_length:]

@models.cml_model
def api_wrapper(args):
  # Pick up args from model api
  prompt = args["prompt"]
  temperature = float(opt_args_value(args, "temperature", 0.7))
  max_new_tokens = float(opt_args_value(args, "max_new_tokens", 50))
  top_p = float(opt_args_value(args, "top_p", 1.0))
  top_k = int(opt_args_value(args, "top_k", 0))
  repetition_penalty = float(opt_args_value(args, "repetition_penalty", 1.0))
  num_beams = int(opt_args_value(args, "num_beams", 1))
  
  return generate(prompt, max_new_tokens, temperature, repetition_penalty, num_beams, top_p, top_k)
