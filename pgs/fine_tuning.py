import streamlit as st 
import pandas as pd 
import numpy as np 
import time
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from setfit.trainer import TrainerCallback, TrainerState
from peft import LoraConfig
import ft
import ft.fine_tune
import os
from ft.app import get_app
from ft.state import get_state

DATA_TEXT_FIELD="prediction"

NUM_EPOCHS = 1.0



TRAINING_PROMPTS = {

    "Clinton/Text-to-sql-v1": 
"""<INSTRUCT>: {instruction}
<INPUT>: {input}
<RESPONSE>: {response}""",

    "hakurei/open-instruct-v1":
"""<INSTRUCTION>: {instruction}
<INPUT>: {input}
<OUTPUT>: {output}"""

}



# # Load a model (HF has caching, so no need to cache this w/ Streamlit)
# def load_base_model(model_name):
#     model = AutoModelForCausalLM.from_pretrained(current_model_name).to("cpu")
#     tokenizer = AutoTokenizer.from_pretrained(current_model_name)
#     return model, tokenizer

# # Loads a dataset (no need to cache because HF caches by default)
# def load_dataset(dataset_name, dataset_fraction = 100):
#     return datasets.load_dataset(dataset_name, split=f'train[:{dataset_fraction}%]')

# # Map a dataset with a new prompt template
# def map_dataset_with_prompt_template(dataset: datasets.Dataset, prompt_template: str):
#     def ds_map(data):
#         data[DATA_TEXT_FIELD] = prompt_template.format(**data)
#         return data
#     dataset = dataset.map(ds_map)
#     return dataset





st.header("Model")
current_models = get_state().models
model_idx = st.selectbox("Models", range(len(current_models)), format_func=lambda x: current_models[x].name, index=None)

st.header("Dataset")
current_datasets = get_state().datasets
dataset_idx = st.selectbox("Dataset", range(len(current_datasets)), format_func=lambda x: current_datasets[x].name, index=None)

if dataset_idx is not None:
    dataset = current_datasets[dataset_idx]

    st.header("Prompt Template")
    current_prompts = get_state().prompts
    current_prompts = list(filter(lambda x: x.dataset_id == dataset.id, current_prompts))
    prompt_idx = st.selectbox("Prompts", range(len(current_prompts)), format_func=lambda x: current_prompts[x].name, index=None)

    if prompt_idx is not None:
        st.text_area("Prompt Template", value=current_prompts[prompt_idx].prompt_template, disabled=True, height=300)
    


# if st.button("Fine Tune Model"):
#     with st.spinner("Mapping Prompt Template to Dataset"):
#         mapped_dataset = map_dataset_with_prompt_template(dataset, prompt_template)

#     with st.spinner("Preparing Trainer..."):

#         # Load a model using PEFT library for finetuning
#         ft = ft.fine_tuner.AMPFineTuner(current_model_name)

#         # Set LoRA training configuration
#         ft.set_lora_config(
#             LoraConfig(
#                     r=16,
#                     lora_alpha=32,
#                     # target_modules=["query_key_value", "xxx"],
#                     lora_dropout=0.05,
#                     bias="none",
#                     task_type="CAUSAL_LM"
#                 )
#         )

#         # Set training arguments 
#         # see fine_tuner.py for list of defaults and huggingface's transformers.TrainingArguments
#         # or the full list of arguments
#         ft.training_args.num_train_epochs=NUM_EPOCHS
#         ft.training_args.warmup_ratio=0.03
#         ft.training_args.max_grad_norm=0.3
#         ft.training_args.learning_rate=2e-4

#     pbar = st.progress(0.0, "Training Progress")

#     class PBarCallback(TrainerCallback):
#         def __init__(self, pbar):
#             self.pbar = pbar
#             return
        
#         def on_log(self, args, state: TrainerState, control, logs=None, **kwargs):
#             epoch = state.epoch
#             completion_amount = epoch / NUM_EPOCHS
#             pbar.progress(completion_amount)


#     # Execute training and save adapter
#     ft.train(mapped_dataset, DATA_TEXT_FIELD, f"adapters/{output_adapter_name}", callbacks=[PBarCallback(pbar)])
