import streamlit as st 
import pandas as pd 
import numpy as np 
import time
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from setfit.trainer import TrainerCallback, TrainerState
from peft import LoraConfig
import ft
import ft.fine_tune
import os
from ft.app import get_app
from ft.state import get_state
from ft.job import StartFineTuningJobRequest, FineTuningJobMetadata, StartFineTuningJobResponse
import json
import torch


with st.container(border=True):
    col1, col2 = st.columns([1,13])
    with col1:
        col1.image("./resources/images/forward_24dp_EA3323_FILL0_wght400_GRAD0_opsz40.png")
    with col2:
        col2.subheader('Train a new Adapater', divider='orange')
        st.write("Finetune your model using the imported datasets and models, leveraging advanced techniques to improve performance.")

with st.container(border=True):
  
    DATA_TEXT_FIELD="prediction"

    NUM_EPOCHS = 1.0


    current_models = get_state().models
    model_idx = st.selectbox("Models", range(len(current_models)), format_func=lambda x: current_models[x].name, index=None)

    current_datasets = get_state().datasets
    dataset_idx = st.selectbox("Dataset", range(len(current_datasets)), format_func=lambda x: current_datasets[x].name, index=None)

    prompt_idx = None

    if dataset_idx is not None:
        dataset = current_datasets[dataset_idx]
        current_prompts = get_state().prompts
        current_prompts = list(filter(lambda x: x.dataset_id == dataset.id, current_prompts))
        prompt_idx = st.selectbox("Prompts", range(len(current_prompts)), format_func=lambda x: current_prompts[x].name, index=None)

        if prompt_idx is not None:
            st.code(current_prompts[prompt_idx].prompt_template)

    start_job_button = None 


    adapter_name = st.text_input("Adapter Name", placeholder="human-friendly adapter name for future reference")

    adapter_location = st.text_input("Output Location", value="data/adapters/")

    advanced_expander = st.expander("Advanced Options")
    c1, c2 = advanced_expander.columns([1,1])
    lora_config = c1.text_area("LoRA Config", json.dumps(json.load(open(".app/configs/default_lora_config.json")), indent=2), height=200)
    bnb_config = c2.text_area("BitsAndBytes Config", json.dumps(json.load(open(".app/configs/default_bnb_config.json")), indent=2), height=200)

    button_enabled = dataset_idx is not None and model_idx is not None and prompt_idx is not None and adapter_name != ""
    start_job_button = st.button("Start Job", type="primary", use_container_width=True, disabled=not button_enabled)

    if start_job_button:
        model = current_models[model_idx]
        dataset = current_datasets[dataset_idx]
        prompt = current_prompts[prompt_idx]
        bnb_config_dict = json.loads(bnb_config)
        get_app().launch_ft_job(StartFineTuningJobRequest(
            adapter_name=adapter_name,
            base_model_id=model.id,
            dataset_id=dataset.id,
            prompt_id=prompt.id,
            num_workers=1,
            bits_and_bytes_config=BitsAndBytesConfig(**bnb_config_dict),
            auto_add_adapter=True,
        ))

    # TODO: start up a job


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
