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
from ft.mlflow import MLflowEvaluationJobMetadata, StartMLflowEvaluationJobRequest, StartMLflowEvaluationJobResponse
import json
import torch
from ft.utils import get_env_variable, fetch_resource_usage_data
from typing import List, Optional, Dict, Any
from ft.adapter import *

project_owner = get_env_variable('PROJECT_OWNER', 'User')
cdsw_api_url = get_env_variable('CDSW_API_URL')
cdsw_api_key = get_env_variable('CDSW_API_KEY')
cdsw_project_url = get_env_variable('CDSW_PROJECT_URL')


def process_resource_usage_data(data: Dict[str, Any]) -> pd.DataFrame:
    """Process the JSON data to extract relevant information and return a DataFrame."""
    user_data = data.get('user', {})
    quotas = user_data.get('quotas', {})

    resources = ['cpu', 'memory', 'nvidiaGPU']
    rows = []

    for resource in resources:
        current_usage = user_data.get(resource, 0)
        if resource == 'nvidiaGPU':
            max_usage = quotas.get('requestsGpu', '0')
            max_usage_value = int(max_usage)
            unit = "GPU"
        elif resource == 'memory':
            max_usage = quotas.get('requestsMemory', '0Gi')
            max_usage_value = float(max_usage.replace('Gi', '').replace('Ti', '')) * (1024 if 'Ti' in max_usage else 1)
            unit = "GiB"
        else:
            max_usage = quotas.get(f'requestsCpu', '0')
            max_usage_value = int(max_usage)
            unit = "vCPU"

        rows.append({
            'Resource Name': resource.upper(),
            'Progress': current_usage / max_usage_value * 100 if max_usage_value else 0,
            'Used': f"{current_usage:.2f} {unit}",
            'Max Available': f"{max_usage_value} {unit}"
        })

    return pd.DataFrame(rows)


# Container for header
with st.container(border=True):
    col1, col2 = st.columns([1, 17])
    with col1:
        col1.image("./resources/images/model_training_24dp_EA3323_FILL0_wght400_GRAD0_opsz48.png")
    with col2:
        col2.subheader('Run MLFlow Evaluation', divider='red')
        st.caption("Execute comprehensive MLFlow evaluations on your fine-tuned model to ensure accuracy, performance, and reliability, gaining valuable insights.")

st.write("\n")

# Container for model and adapter selection
ccol1, ccol2 = st.columns([3, 2])
with ccol1:
    with st.container(border=True):

        CURRENT_MODEL = None

        current_models = get_state().models
        model_idx = st.selectbox(
            "Base Models",
            range(len(current_models)),
            format_func=lambda x: current_models[x].name,
            index=None
        )

        model_adapter_idx = None

        # TODO: this currently assumes HF model for local eval, but should not have to be in the future
        if model_idx is not None:
            current_model_metadata = current_models[model_idx]

            model_adapters: List[AdapterMetadata] = get_state().adapters
            model_adapters = list(filter(lambda x: x.model_id == current_model_metadata.id, model_adapters))

            # Filter adapters based on their presence in the /data/adapter directory
            model_adapters = list(filter(lambda x: os.path.isdir(os.path.join(x.location)), model_adapters))

            # TODO: We should not have to load the adapters every run, this is overkill
            with st.spinner("Loading Adapters..."):
                for adapter in model_adapters:
                    loc = os.path.join(adapter.location)
                    if not loc.endswith("/"):
                        loc += "/"

            model_adapter_idx = st.selectbox(
                "Choose an Adapter",
                range(len(model_adapters)),
                format_func=lambda x: model_adapters[x].name,
                index=None
            )

            if model_adapter_idx is not None:
                model_adapter = model_adapters[model_adapter_idx]

            # Container for dataset and prompt selection
            col1, col2 = st.columns(2)

        current_datasets = get_state().datasets
        dataset_idx = st.selectbox(
            "Datasets",
            range(len(current_datasets)),
            format_func=lambda x: current_datasets[x].name,
            index=None
        )

        # Advanced options
        st.markdown("---")
        st.caption("**Advance Options**")
        c1, c2= st.columns([1, 1])
        with c1:
            cpu = st.text_input("CPU(vCPU)", value="2", key="cpu")
        with c2:
            memory = st.text_input("Memory(GiB)", value="8", key="memory")

        gpu = st.selectbox("GPU(NVIDIA)", options=[1], index=0)

        button_enabled = dataset_idx is not None and model_idx is not None and model_adapter_idx is not None
        start_job_button = st.button(
            "Start MLflow Evaluation Job",
            type="primary",
            use_container_width=True,
            disabled=not button_enabled)

        if start_job_button:
            try:
                model = current_models[model_idx]
                dataset = current_datasets[dataset_idx]
                adapter = model_adapters[model_adapter_idx]
                print(model.id)
                print(dataset.id)
                print(adapter.id)
                get_app().launch_mlflow_job(StartMLflowEvaluationJobRequest(
                    adapter_id=adapter.id,
                    base_model_id=model.id,
                    dataset_id=dataset.id,
                    cpu=int(cpu),
                    gpu=gpu,
                    memory=int(memory)
                ))
                st.success("Created MLflow Job. Please go to **View MLflow Runs** tab!", icon=":material/check:")
                st.toast("Created MLflow Job. Please go to **View MLflow Runs** tab!", icon=":material/check:")
            except Exception as e:
                st.error(f"Failed to create MLflow Job: **{str(e)}**", icon=":material/error:")
                st.toast(f"Failed to create MLflow Job: **{str(e)}**", icon=":material/error:")

with ccol2:
    st.info(
        """
        This page allows you to run MLflow evaluation jobs on your fine-tuned models and their corresponding adapters.
        The evaluation job will generate a detailed report on the performance of the adapter using a sample dataset.
        You can view the complete evaluation report on the **View MLflow Runs** page.
        """,
        icon=":material/info:"
    )
    st.write("\n")
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
                "Max Available": "User Quota"
            },
            hide_index=True,
            use_container_width=True
        )
