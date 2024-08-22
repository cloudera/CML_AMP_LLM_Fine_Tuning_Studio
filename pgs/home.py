import streamlit as st
from ft.utils import get_env_variable, fetch_resource_usage_data, process_resource_usage_data
from typing import List
import pandas as pd
import requests
from ft.api import *
from google.protobuf.json_format import MessageToDict
from pgs.streamlit_utils import get_fine_tuning_studio_client
import json

# Instantiate the client to the FTS gRPC app server.
fts = get_fine_tuning_studio_client()


def create_homepage_header():
    with st.container(border=True):
        col1, col2 = st.columns([1, 16])
        with col1:
            col1.image("./resources/images/architecture_24dp_EA3323_FILL0_wght400_GRAD0_opsz48.png")
        with col2:
            col2.subheader('LLM Finetuning Studio', divider='red')
            col2.caption(
                'The LLM Fine Tuning Studio, updated in July 2024, features a new Streamlit-based UI and integrates with Cloudera Machine Learning (CML) components. '
                'It supports custom datasets, BitsAndBytes, LoRA configurations, and distributed training.')


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


# st.subheader(f"Welcome to LLM Finetuning Studio, {project_owner}", divider="red")
create_homepage_header()
# st.write("\n")

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

datasets: List[DatasetMetadata] = fts.get_datasets()
current_jobs = fts.get_fine_tuning_jobs()
current_adapters = fts.get_adapters()

col1, col2, col3 = st.columns([5, 4, 4])
with col1:
    col1.caption("**Resource Usage**")
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

with col2:
    st.caption("**Training Jobs**")
    if current_jobs:
        jobs_df = pd.DataFrame([MessageToDict(res, preserving_proto_field_name=True) for res in current_jobs])
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
                        how='inner',
                        suffixes=('', '_cml'))

                    display_df = display_df[['id', 'num_workers', 'latest']]

                    status_mapping = {
                        "succeeded": 100,
                        "running": 30,
                        "scheduling": 1
                    }
                    display_df['status'] = display_df['latest'].apply(
                        lambda x: status_mapping.get(x['status'], 0) if pd.notnull(x) else 0)

                    display_df['adapter_name'] = display_df['id'].map(lambda x: next(
                        (adapter.name for adapter in current_adapters if adapter.fine_tuning_job_id == x), "Unknown"))

                    st.data_editor(
                        display_df[['id', 'status', 'adapter_name']],
                        column_config={
                            "id": st.column_config.TextColumn("Job ID"),
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
        jobs_df = pd.DataFrame(columns=['id', 'status', 'adapter_name'])
        st.data_editor(
            jobs_df[['id', 'status', 'adapter_name']],
            column_config={
                "id": st.column_config.TextColumn("Job ID"),
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

with col3:
    st.caption("**MLflow Jobs**")
    current_jobs = fts.get_evaluation_jobs()
    if current_jobs:
        jobs_df = pd.DataFrame([MessageToDict(res, preserving_proto_field_name=True) for res in current_jobs])
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
                        how='inner',
                        suffixes=('', '_cml'))

                    display_df = display_df[['id', 'num_workers', 'latest']]

                    status_mapping = {
                        "succeeded": 100,
                        "running": 30,
                        "scheduling": 1
                    }
                    display_df['status'] = display_df['latest'].apply(
                        lambda x: status_mapping.get(x['status'], 0) if pd.notnull(x) else 0)

                    display_df['adapter_name'] = display_df['id'].map(lambda x: next(
                        (adapter.name for adapter in current_adapters if adapter.fine_tuning_job_id == x), "Unknown"))

                    st.data_editor(
                        display_df[['job_id', 'status']],
                        column_config={
                            "job_id": st.column_config.TextColumn("Job ID"),
                            "status": st.column_config.ProgressColumn(
                                "Status",
                                help="Job status as progress",
                                format="%.0f%%",
                                min_value=0,
                                max_value=100,
                            )
                        },
                        hide_index=True,
                        use_container_width=True,
                        height=140
                    )
            except requests.RequestException as e:
                st.error(f"Failed to fetch jobs from API: {e}")
    else:
        jobs_df = pd.DataFrame(columns=['job_id', 'status'])
        st.data_editor(
            jobs_df[['job_id', 'status']],
            column_config={
                "job_id": st.column_config.TextColumn("Job ID"),
                "status": st.column_config.ProgressColumn(
                    "Status",
                    help="Job status as progress",
                    format="%d",
                    min_value=0,
                    max_value=100,
                )
            },
            hide_index=True,
            use_container_width=True
        )

st.write("\n")

col1, col2 = st.columns([1, 1])

with col1:
    st.caption("**Datasets**")
    data_dicts = [{"name": dataset.name, "features": json.loads(dataset.features)} for dataset in datasets]
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

with col2:
    st.caption("**Models & Adapters**")
    models: List[ModelMetadata] = fts.get_models()
    adapters: List[AdapterMetadata] = fts.get_adapters()

    # Prepare data for the data editor
    data = []
    if models:
        for model in models:
            model_adapters = [adapter.name for adapter in adapters if adapter.model_id == model.id]
            if len(model_adapters) > 0:
                data.append({"Model": model.name, "Adapters": model_adapters})
            else:
                data.append({"Model": model.name, "Adapters": [" -- No Adapters available for this base model -- "]})
    else:
        data.append({"Model": "", "Adapters": ""})

    # Display the data editor
    st.data_editor(data, use_container_width=True)
