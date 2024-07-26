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

# Container for header
with st.container(border=True):
    col1, col2 = st.columns([1, 17])
    with col1:
        st.image("./resources/images/forward_24dp_EA3323_FILL0_wght400_GRAD0_opsz40.png")
    with col2:
        st.subheader('Train a new Adapter', divider='red')
        st.caption("Finetune your model using the imported datasets and models, leveraging advanced techniques to improve performance.")

# Container for model and adapter selection
with st.container(border=True):
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            adapter_name = st.text_input("Adapter Name", placeholder="human-friendly adapter name for future reference", key="adapter_name")

        with col2:
            current_models = get_state().models
            model_idx = st.selectbox("Models", range(len(current_models)), format_func=lambda x: current_models[x].name, index=None)

    # Container for dataset and prompt selection
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            current_datasets = get_state().datasets
            dataset_idx = st.selectbox("Dataset", range(len(current_datasets)), format_func=lambda x: current_datasets[x].name, index=None)
            if dataset_idx is not None:
                dataset = current_datasets[dataset_idx]
                current_prompts = get_state().prompts
                current_prompts = list(filter(lambda x: x.dataset_id == dataset.id, current_prompts))
                prompt_idx = st.selectbox("Prompts", range(len(current_prompts)), format_func=lambda x: current_prompts[x].name, index=None)

                if prompt_idx is not None:
                    st.code(current_prompts[prompt_idx].prompt_template)

        with col2:
            adapter_location = st.text_input("Output Location", value="data/adapters/", key="output_location")

    # Advanced options
    with st.expander("Advanced Options", expanded=True):
        c1, c2 = st.columns([1, 1])
        lora_config = c1.text_area("LoRA Config", json.dumps(json.load(open(".app/configs/default_lora_config.json")), indent=2), height=200)
        bnb_config = c2.text_area("BitsAndBytes Config", json.dumps(json.load(open(".app/configs/default_bnb_config.json")), indent=2), height=200)

    # Start job button
    button_enabled = dataset_idx is not None and model_idx is not None and prompt_idx is not None and adapter_name != ""
    start_job_button = st.button("Start Job", type="primary", use_container_width=True, disabled=not button_enabled)

    if start_job_button:
        try:
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
            st.success("Create Finetuning Job. Please go to **Monitor Training Job** tab!", icon=":material/check:")
            st.toast("Create Finetuning Job. Please go to **Monitor Training Job** tab!", icon=":material/check:")
        except Exception as e:
            st.error(f"Failed to create Finetuning Job: **{str(e)}**", icon=":material/error:")
            st.toast(f"Failed to Finetuning Job: **{str(e)}**", icon=":material/error:")
