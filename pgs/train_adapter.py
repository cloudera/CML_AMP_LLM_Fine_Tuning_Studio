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
from ft.utils import get_env_variable, fetch_resource_usage_data, process_resource_usage_data
from typing import List, Optional, Dict, Any

cdsw_api_url = get_env_variable('CDSW_API_URL')
cdsw_api_key = get_env_variable('CDSW_API_KEY')
cdsw_project_url = get_env_variable('CDSW_PROJECT_URL')
project_owner = get_env_variable('PROJECT_OWNER', 'User')


# Container for header
def create_header():
    with st.container(border=True):
        col1, col2 = st.columns([1, 17])
        with col1:
            st.image("./resources/images/forward_24dp_EA3323_FILL0_wght400_GRAD0_opsz40.png")
        with col2:
            st.subheader('Train a new Adapter', divider='red')
            st.caption(
                "Finetune your model using the imported datasets and models, leveraging advanced techniques to improve performance.")

# Container for model and adapter selection


def create_train_adapter_page():
    ccol1, ccol2 = st.columns([3, 2])
    with ccol1:
        with st.container(border=True):

            col1, col2 = st.columns(2)
            with col1:
                adapter_name = st.text_input("Adapter Name", placeholder="Adapter name", key="adapter_name")

            with col2:
                current_models = get_state().models
                model_idx = st.selectbox(
                    "Base Models",
                    range(
                        len(current_models)),
                    format_func=lambda x: current_models[x].name,
                    index=None)

            # Container for dataset and prompt selection
            col1, col2 = st.columns(2)

            with col1:
                current_datasets = get_state().datasets
                dataset_idx = st.selectbox(
                    "Dataset",
                    range(
                        len(current_datasets)),
                    format_func=lambda x: current_datasets[x].name,
                    index=None)
                if dataset_idx is not None:
                    dataset = current_datasets[dataset_idx]
                    current_prompts = get_state().prompts
                    current_prompts = list(filter(lambda x: x.dataset_id == dataset.id, current_prompts))
                    prompt_idx = st.selectbox(
                        "Prompts",
                        range(
                            len(current_prompts)),
                        format_func=lambda x: current_prompts[x].name,
                        index=None)

                    if prompt_idx is not None:
                        st.code(current_prompts[prompt_idx].prompt_template)

            with col2:
                adapter_location = st.text_input("Output Location", value="data/adapters/", key="output_location")

            # Advanced options
            c1, c2 = st.columns(2)
            with c1:
                num_epochs = st.text_input("Number of Epochs", value="10", key="num_epochs")
            with c2:
                learning_rate = st.text_input("Learning Rate", value="2e-4", key="learning_rate")
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                cpu = st.text_input("CPU (vCPU)", value="2", key="cpu")
            with c2:
                memory = st.text_input("Memory (GiB)", value="8", key="memory")
            with c3:
                gpu = st.selectbox("GPU (NVIDIA)", options=[1], index=0)

            c1, c2 = st.columns([1, 1])
            lora_config = c1.text_area(
                "LoRA Config",
                json.dumps(
                    json.load(
                        open(".app/configs/default_lora_config.json")),
                    indent=2),
                height=200)
            bnb_config = c2.text_area(
                "BitsAndBytes Config",
                json.dumps(
                    json.load(
                        open(".app/configs/default_bnb_config.json")),
                    indent=2),
                height=200)

            # Start job button
            button_enabled = dataset_idx is not None and model_idx is not None and prompt_idx is not None and adapter_name != ""
            start_job_button = st.button(
                "Start Job",
                type="primary",
                use_container_width=True,
                disabled=not button_enabled)

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
                        num_epochs=int(num_epochs),
                        learning_rate=float(learning_rate),
                        cpu=int(cpu),
                        gpu=gpu,
                        memory=int(memory)
                    ))
                    st.success(
                        "Create Finetuning Job. Please go to **Monitor Training Job** tab!",
                        icon=":material/check:")
                    st.toast(
                        "Create Finetuning Job. Please go to **Monitor Training Job** tab!",
                        icon=":material/check:")
                except Exception as e:
                    st.error(f"Failed to create Finetuning Job: **{str(e)}**", icon=":material/error:")
                    st.toast(f"Failed to Finetuning Job: **{str(e)}**", icon=":material/error:")

    with ccol2:
        st.info("""
        ### Advanced Options

        - **CPU**: Enter the number of CPU cores to be used for training.
        - **Memory**: Specify the amount of memory (in GiB) to be allocated for the training process.
        - **GPU**: Select the number of GPUs to be used. Currently, only one GPU option is available.
        - **LoRA Config**: Provide the configuration for LoRA (Low-Rank Adaptation), which is a technique used to fine-tune models with fewer parameters.
        - **BitsAndBytes Config**: Provide the configuration for the BitsAndBytes library, which can optimize model performance during training.

        ### Starting the Job

        The button will be enabled only when all required fields are filled out.
        """)
        ccol2.caption("**Resource Usage**")
        data = fetch_resource_usage_data(cdsw_api_url, project_owner, cdsw_api_key)
        if data:
            df = process_resource_usage_data(data)
            st.data_editor(
                df[['Resource Name', 'Progress', 'Max Available']],
                column_config={
                    "Resource Name": "Resource Name",
                    "Progress": st.column_config.ProgressColumn(
                        "Usage",
                        help="Current Resource Consumption",
                        format="%.0f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "Max Available": "Available (Cluster)"
                },
                hide_index=True,
                use_container_width=True
            )


create_header()
create_train_adapter_page()
