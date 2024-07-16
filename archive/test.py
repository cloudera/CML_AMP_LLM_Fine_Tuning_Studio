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

DATA_TEXT_FIELD="prediction"

NUM_EPOCHS = 1.0







# Load a model (HF has caching, so no need to cache this w/ Streamlit)
def load_base_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(current_model_name).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(current_model_name)
    return model, tokenizer

# Loads a dataset (no need to cache because HF caches by default)
def load_dataset(dataset_name, dataset_fraction = 100):
    return datasets.load_dataset(dataset_name, split=f'train[:{dataset_fraction}%]')

# Map a dataset with a new prompt template
def map_dataset_with_prompt_template(dataset: datasets.Dataset, prompt_template: str):
    def ds_map(data):
        data[DATA_TEXT_FIELD] = prompt_template.format(**data)
        return data
    dataset = dataset.map(ds_map)
    return dataset












st.header("Model")
current_model_name = st.selectbox("Select a Model", AVAILABLE_MODELS)
# with st.spinner("Loading model..."):
#     model, tokenizer = load_base_model(current_model_name)


st.header("Dataset")
current_dataset_name = st.selectbox("Select a Dataset", AVAILABLE_DATASETS)
with st.spinner("Loading Dataset..."):
    dataset = load_dataset(current_dataset_name)
c = st.container()
c.text("Dataset Features:")
for col in dataset.column_names:
    cc = c.container(border=True)
    cc.text(col)


# TODO: make important in demo
st.header("Training Prompt")
prompt_template = st.text_area("Prompt Template", TRAINING_PROMPTS[current_dataset_name])
st.text_area("Example Prompt", prompt_template.format(**dataset[0]), disabled=True)


st.header("Fine Tuning")

output_adapter_name = st.text_input("Adapter name", "custom_adapter")

if st.button("Fine Tune Model"):
    with st.spinner("Mapping Prompt Template to Dataset"):
        mapped_dataset = map_dataset_with_prompt_template(dataset, prompt_template)

    with st.spinner("Preparing Trainer..."):

        # Load a model using PEFT library for finetuning
        ft = ft.fine_tune.AMPFineTuner(current_model_name)

        # Set LoRA training configuration
        ft.set_lora_config(
            LoraConfig(
                    r=16,
                    lora_alpha=32,
                    # target_modules=["query_key_value", "xxx"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
        )

        # Set training arguments 
        # see fine_tuner.py for list of defaults and huggingface's transformers.TrainingArguments
        # or the full list of arguments
        ft.training_args.num_train_epochs=NUM_EPOCHS
        ft.training_args.warmup_ratio=0.03
        ft.training_args.max_grad_norm=0.3
        ft.training_args.learning_rate=2e-4

    pbar = st.progress(0.0, "Training Progress")

    class PBarCallback(TrainerCallback):
        def __init__(self, pbar):
            self.pbar = pbar
            return
        
        def on_log(self, args, state: TrainerState, control, logs=None, **kwargs):
            epoch = state.epoch
            completion_amount = epoch / NUM_EPOCHS
            pbar.progress(completion_amount)


    # Execute training and save adapter
    ft.train(mapped_dataset, DATA_TEXT_FIELD, f"adapters/{output_adapter_name}", callbacks=[PBarCallback(pbar)])
