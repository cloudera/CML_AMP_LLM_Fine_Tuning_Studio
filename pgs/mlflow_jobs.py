import streamlit as st
import pandas as pd
import os
import requests
import json
from google.protobuf.json_format import MessageToDict
from pgs.streamlit_utils import get_fine_tuning_studio_client

from ft.utils import format_status_with_icon

# Instantiate the client to the FTS gRPC app server.
fts = get_fine_tuning_studio_client()


# Container for the layout
with st.container(border=True):
    col1, col2 = st.columns([1, 17])
    with col1:
        col1.image("./resources/images/monitoring_24dp_EA3323_FILL0_wght400_GRAD0_opsz48.png")
    with col2:
        col2.subheader('View MLflow Runs')
        st.caption("Examine the MLflow evaluation results for adapters trained on foundation models, specifically for the datasets they were trained on.")

st.write("\n\n")

# Fetch current jobs from state
current_jobs = fts.get_evaluation_jobs()
models = fts.get_models()
adapters = fts.get_adapters()
datasets = fts.get_datasets()

# Create dictionaries for ID to name mapping
model_dict = {model.id: model.name for model in models}
adapter_dict = {adapter.id: adapter.name for adapter in adapters}
dataset_dict = {dataset.id: dataset.name for dataset in datasets}

# Check if there are any current jobs
tab1, tab2 = st.tabs(["**MLflow Jobs List**", "**View MLflow Runs**"])

with tab1:
    if not current_jobs:
        st.info("No MLflow jobs triggered.", icon=":material/info:")
    else:
        # Convert jobs to DataFrame
        try:
            jobs_df = pd.DataFrame([MessageToDict(res, preserving_proto_field_name=True) for res in current_jobs])
        except Exception as e:
            st.error(f"Error converting jobs to DataFrame: {e}")
            jobs_df = pd.DataFrame()

        # Check if 'cml_job_id' exists in jobs_df
        if 'cml_job_id' not in jobs_df.columns:
            st.error("Column 'cml_job_id' not found in jobs_df")
        else:
            # Use APIv1 for jobs to get id and URL for each job
            HOST = os.getenv('CDSW_PROJECT_URL')
            API_KEY = os.getenv('CDSW_API_KEY')

            if not HOST or not API_KEY:
                st.error("Environment variables for CDSW_PROJECT_URL or CDSW_API_KEY are missing.")
            else:
                url = "/".join([HOST, "jobs"])
                try:
                    res = requests.get(
                        url,
                        headers={"Content-Type": "application/json"},
                        auth=(API_KEY, "")
                    )
                    res.raise_for_status()
                except requests.RequestException as e:
                    st.error(f"Failed to fetch jobs from API: {e}")
                else:
                    # Convert API response to DataFrame
                    try:
                        cml_jobs_list = res.json()
                        cml_jobs_list_df = pd.DataFrame(cml_jobs_list)
                    except (json.JSONDecodeError, ValueError) as e:
                        st.error(f"Error decoding API response: {e}")
                        cml_jobs_list_df = pd.DataFrame()

                    # Check if 'public_identifier' exists in cml_jobs_list_df
                    if 'public_identifier' not in cml_jobs_list_df.columns:
                        st.error("Column 'public_identifier' not found in cml_jobs_list_df")
                    else:
                        # Merge the DataFrames for app state jobs and apiv1 cml jobs
                        display_df = pd.merge(
                            jobs_df,
                            cml_jobs_list_df,
                            left_on='cml_job_id',
                            right_on='public_identifier',
                            suffixes=('', '_cml'))

                        # Replace IDs with names using the dictionaries
                        display_df['adapter_name'] = display_df['adapter_id'].map(adapter_dict)
                        display_df['base_model_name'] = display_df['base_model_id'].map(model_dict)
                        display_df['dataset_name'] = display_df['dataset_id'].map(dataset_dict)

                        # Filter for only columns we care about
                        display_df = display_df[['id', 'html_url', 'latest',
                                                'adapter_name', 'base_model_name', 'dataset_name']]

                        display_df['Status'] = display_df['Status'].apply(
                            lambda x: x['status'] if isinstance(
                                x, dict) and 'status' in x else 'Unknown')
                        display_df['status_with_icon'] = display_df['Status'].apply(format_status_with_icon)

                        # Display the grid with the merged and filtered dataframe
                        st.data_editor(
                            display_df[["id", "Status", "html_url", "adapter_name", "base_model_name", "dataset_name"]],
                            column_config={
                                "id": st.column_config.TextColumn("Job ID"),
                                "status_with_icon": st.column_config.TextColumn(
                                    "Status",
                                    help="Job status as text with icon",
                                ),
                                "html_url": st.column_config.LinkColumn(
                                    "Job Url", display_text="Open CML Job"
                                ),
                                "adapter_name": st.column_config.TextColumn("Adapter"),
                                "base_model_name": st.column_config.TextColumn("Base Model"),
                                "dataset_name": st.column_config.TextColumn("Dataset")
                            },
                            height=540,
                            hide_index=True,
                            use_container_width=True
                        )

with tab2:
    if not current_jobs:
        st.info("No MLflow jobs triggered.", icon=":material/info:")
    else:
        # Extract cml_job_ids
        job_ids = [job.id for job in current_jobs]

        # Select a cml_job_id from the list
        selected_job_id = st.selectbox('Select Job ID', job_ids, index=0)
        st.write("\n")
        st.caption("**Evaluation Results**")

        # Find the job corresponding to the selected job_id
        selected_job = next((job for job in current_jobs if job.id == selected_job_id), None)

        if selected_job:
            # Get the evaluation CSV file path for the selected job
            csv_file_path = os.path.join(selected_job.evaluation_dir, 'result_evaluation.csv')

            # Read the CSV file
            if os.path.exists(csv_file_path):
                df = pd.read_csv(csv_file_path)
                # Display the dataframe in Streamlit
                st.data_editor(df, hide_index=True)
            else:
                st.info(
                    "**Evaluation report** is not available yet. Please wait for MLflow run to complete.",
                    icon=':material/info:')
        else:
            st.error("Selected job not found.")
