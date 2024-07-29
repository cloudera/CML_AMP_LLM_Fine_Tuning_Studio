import streamlit as st
from ft.dataset import DatasetMetadata, DatasetType
from ft.app import get_app
from ft.state import get_state
from ft.utils import get_env_variable, fetch_resource_usage_data
from typing import List, Optional, Dict, Any
import pandas as pd
import os
import requests


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


def create_tile(container, image_path: str, button_text: str, page_path: str, description: str) -> None:
    """Create a tile with an image, button, and description."""
    tile = container.container(height=160)
    if tile.button(button_text, type="primary", use_container_width=True):
        st.switch_page(page_path)
    c1, c2 = tile.columns([1, 4])
    with c1:
        c1.image(image_path)
    with c2:
        c2.caption(description)


project_owner = get_env_variable('PROJECT_OWNER', 'User')
cdsw_api_url = get_env_variable('CDSW_API_URL')
cdsw_api_key = get_env_variable('CDSW_API_KEY')
cdsw_project_url = get_env_variable('CDSW_PROJECT_URL')

st.subheader(f"Welcome, {project_owner}")

col1, col2, col3, col4 = st.columns(4)
create_tile(
    col1,
    "./resources/images/publish_24dp_EA3323_FILL0_wght400_GRAD0_opsz48.png",
    "Import Datasets",
    "pgs/datasets.py",
    'Import datasets from Hugging Face or upload your own preprocessed dataset from local sources for fine-tuning.')

create_tile(
    col2,
    "./resources/images/neurology_24dp_EA3323_FILL0_wght400_GRAD0_opsz40.png",
    "Import Base Models",
    "pgs/models.py",
    'Import foundational LLM models from Hugging Face or local sources for your fine-tuning job specific requirements.')

create_tile(col3, "./resources/images/forward_24dp_EA3323_FILL0_wght400_GRAD0_opsz40.png",
            "Finetune your model", "pgs/train_adapter.py",
            'Finetune your model, leveraging advanced techniques to improve performance.')

create_tile(col4, "./resources/images/move_group_24dp_EA3323_FILL0_wght400_GRAD0_opsz40.png",
            "Export to CML Model Registry", "pgs/export.py",
            'Export your fine-tuned models and adapters to the Cloudera Model Registry for easy access and deployment.')

st.write("\n")

datasets: List[DatasetMetadata] = get_app().datasets.list_datasets()
current_jobs = get_state().jobs
current_adapters = get_state().adapters

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    col1.caption("**Resource Usage**")
    data = fetch_resource_usage_data(cdsw_api_url, cdsw_api_key)
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
                "Max Available": "Max Available"
            },
            hide_index=True,
            use_container_width=True
        )

with col2:
    st.caption("**Jobs & Adapters**")
    if current_jobs:
        jobs_df = pd.DataFrame([res.model_dump() for res in current_jobs])
        if 'cml_job_id' not in jobs_df.columns:
            st.error("Column 'cml_job_id' not found in jobs_df")
        else:
            url = f"{cdsw_project_url}/jobs"
            try:
                res = requests.get(url, headers={"Content-Type": "application/json"}, auth=(cdsw_api_key, ""))
                res.raise_for_status()
                cml_jobs_list = res.json()
                cml_jobs_list_df = pd.DataFrame(cml_jobs_list)

                if 'public_identifier' not in cml_jobs_list_df.columns:
                    st.error("Column 'public_identifier' not found in cml_jobs_list_df")
                else:
                    display_df = pd.merge(
                        jobs_df,
                        cml_jobs_list_df,
                        left_on='cml_job_id',
                        right_on='public_identifier',
                        how='inner')

                    display_df = display_df[['job_id', 'num_workers', 'latest']]

                    status_mapping = {
                        "succeeded": 100,
                        "running": 30,
                        "scheduling": 1
                    }
                    display_df['status'] = display_df['latest'].apply(
                        lambda x: status_mapping.get(x['status'], 0) if pd.notnull(x) else 0)

                    display_df['adapter_name'] = display_df['job_id'].map(
                        lambda x: next((adapter.name for adapter in current_adapters if adapter.job_id == x), "Unknown")
                    )

                    st.data_editor(
                        display_df[['job_id', 'status', 'adapter_name']],
                        column_config={
                            "job_id": st.column_config.TextColumn("Job ID"),
                            "status": st.column_config.ProgressColumn(
                                "Status",
                                help="Job status as progress",
                                format="%.0f%%",
                                min_value=0,
                                max_value=100,
                            ),
                            "adapter_name": st.column_config.TextColumn("Adapter Name")
                        },
                        hide_index=True,
                        use_container_width=True,
                        height=140
                    )
            except requests.RequestException as e:
                st.error(f"Failed to fetch jobs from API: {e}")
    else:
        jobs_df = pd.DataFrame(columns=['job_id', 'status', 'adapter_name'])
        st.data_editor(
            jobs_df[['job_id', 'status', 'adapter_name']],
            column_config={
                "job_id": st.column_config.TextColumn("Job ID"),
                "status": st.column_config.ProgressColumn(
                    "Status",
                    help="Job status as progress",
                    format="%d",
                    min_value=0,
                    max_value=100,
                ),
                "adapter_name": st.column_config.TextColumn("Adapter Name")
            },
            hide_index=True,
            use_container_width=True
        )

st.write("\n")

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.caption("**Datasets**")
    data_dicts = [{"name": dataset.name, "features": dataset.features} for dataset in datasets]
    df = pd.DataFrame(data_dicts)
    st.data_editor(
        df,
        column_config={
            "name": "Name",
            "features": "Features",
        },
        hide_index=True,
        use_container_width=True
    )
