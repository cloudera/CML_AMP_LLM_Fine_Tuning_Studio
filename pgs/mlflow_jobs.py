import streamlit as st
import pandas as pd
import os
import requests
from google.protobuf.json_format import MessageToDict
from pgs.streamlit_utils import get_fine_tuning_studio_client
from ft.utils import format_status_with_icon

# Instantiate the client to the FTS gRPC app server.
fts = get_fine_tuning_studio_client()


def display_page_header():
    with st.container(border=True):
        col1, col2 = st.columns([1, 17])
        with col1:
            col1.image("./resources/images/monitoring_24dp_EA3323_FILL0_wght400_GRAD0_opsz48.png")
        with col2:
            col2.subheader('View MLflow Runs')
            st.caption("Examine the MLflow evaluation results for adapters trained on foundation models.")


def fetch_job_data():
    try:
        current_jobs = fts.get_evaluation_jobs()
        models = fts.get_models()
        adapters = fts.get_adapters()
        datasets = fts.get_datasets()

        model_dict = {model.id: model.name for model in models}
        adapter_dict = {adapter.id: adapter.name for adapter in adapters}
        dataset_dict = {dataset.id: dataset.name for dataset in datasets}

        return current_jobs, model_dict, adapter_dict, dataset_dict
    except Exception as e:
        st.error(f"Error fetching job data: {e}")
        return [], {}, {}, {}


def fetch_api_jobs():
    HOST = os.getenv('CDSW_PROJECT_URL')
    API_KEY = os.getenv('CDSW_API_KEY')

    if not HOST or not API_KEY:
        st.error("Environment variables for CDSW_PROJECT_URL or CDSW_API_KEY are missing.")
        return pd.DataFrame()

    url = "/".join([HOST, "jobs"])
    try:
        res = requests.get(url, headers={"Content-Type": "application/json"}, auth=(API_KEY, ""))
        res.raise_for_status()
        return pd.DataFrame(res.json())
    except requests.RequestException as e:
        st.error(f"Failed to fetch jobs from API: {e}")
        return pd.DataFrame()


@st.fragment
def display_jobs_list():
    current_jobs, model_dict, adapter_dict, dataset_dict = fetch_job_data()

    if not current_jobs:
        st.info("No MLflow jobs triggered.", icon=":material/info:")
        return

    st.write("\n")

    _, col1 = st.columns([14, 2])

    with col1:
        if st.button("Refresh", use_container_width=True, type='primary'):
            st.rerun(scope="fragment")

    # delete_button = col2.button("Delete Jobs", type="primary", use_container_width=True)

    try:
        jobs_df = pd.DataFrame([MessageToDict(res, preserving_proto_field_name=True) for res in current_jobs])
    except Exception as e:
        st.error(f"Error converting jobs to DataFrame: {e}")
        jobs_df = pd.DataFrame()

    if 'cml_job_id' not in jobs_df.columns:
        st.error("Column 'cml_job_id' not found in jobs_df")
        return

    cml_jobs_list_df = fetch_api_jobs()

    if 'public_identifier' not in cml_jobs_list_df.columns:
        st.error("Column 'public_identifier' not found in API job list.")
        return

    display_df = pd.merge(
        jobs_df,
        cml_jobs_list_df,
        left_on='cml_job_id',
        right_on='public_identifier',
        suffixes=('', '_cml')
    )

    display_df['adapter_name'] = display_df['adapter_id'].map(adapter_dict)
    display_df['base_model_name'] = display_df['base_model_id'].map(model_dict)
    display_df['dataset_name'] = display_df['dataset_id'].map(dataset_dict)

    columns_we_care_about = [
        'id',
        'html_url',
        'latest',
        'base_model_name',
        'dataset_name',
        'adapter_name',
        'created_at'
    ]

    for column in columns_we_care_about:
        if column not in display_df.columns:
            display_df[column] = 'Unknown'

    display_df = display_df[columns_we_care_about]

    display_df.rename(columns={
        'id': 'Job ID',
        'base_model_name': 'Model Name',
        'dataset_name': 'Dataset Name',
        'adapter_name': 'Adapter Name',
        'latest': 'Status'
    }, inplace=True)

    display_df['Status'] = display_df['Status'].apply(
        lambda x: x['status'] if isinstance(x, dict) and 'status' in x else 'Unknown')
    display_df['status_with_icon'] = display_df['Status'].apply(format_status_with_icon)

    # display_df["Select"] = False

    # Converting the 'created_at' column from a string in the format '2024-08-27T10:45:38.900Z' to a datetime object
    display_df['created_at'] = pd.to_datetime(display_df['created_at'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    display_df = display_df.sort_values(by='created_at', ascending=False)

    # Data editor for job table
    edited_df = st.data_editor(
        display_df[["Job ID", "status_with_icon", "created_at",
                    "html_url", "Model Name", "Dataset Name", "Adapter Name"]],
        column_config={
            "Job ID": st.column_config.TextColumn("Job ID"),
            "status_with_icon": st.column_config.TextColumn(
                "Status",
                help="Job status as text with icon",
            ),
            "html_url": st.column_config.LinkColumn(
                "CML Job Link", display_text="Open CML Job"
            ),
            "Model Name": st.column_config.TextColumn("Model Name"),
            "Dataset Name": st.column_config.TextColumn("Dataset Name"),
            "Adapter Name": st.column_config.TextColumn("Adapter Name"),
            # "Select": st.column_config.CheckboxColumn("", width="small"),
            "created_at": st.column_config.DatetimeColumn(
                "Created At",
                format="D MMM YYYY, h:mm a",
                step=60,
            )
        },
        height=500,
        hide_index=True,
        use_container_width=True
    )

    # if delete_button:
    #     # Check if edited_df is not empty and contains the "Select" column
    #     if edited_df.empty or "Select" not in edited_df.columns:
    #         st.warning("No jobs available for deletion.")
    #     else:
    #         # Filter selected jobs
    #         selected_jobs = edited_df[edited_df["Select"]]["Job ID"]

    #         if not selected_jobs.empty:
    #             st.toast(f"Deleting jobs: {', '.join(selected_jobs)}")
    #             # Implement your job deletion logic here
    #             for job_id in selected_jobs:
    #                 try:
    #                     response = fts.RemoveEvaluationJob(RemoveEvaluationJobRequest(
    #                         id=job_id
    #                     ))
    #                     st.toast(f"Job {job_id} deleted successfully.")
    #                 except Exception as e:
    #                     st.error(f"Error deleting job {job_id}: {str(e)}")

    #             # After all deletions, reload the specific component or data
    #             st.rerun(scope="fragment")
    #         else:
    #             st.warning("No jobs selected for deletion.")


def display_mlflow_runs():
    current_jobs, model_dict, adapter_dict, dataset_dict = fetch_job_data()
    if not current_jobs:
        st.info("No MLflow jobs triggered.", icon=":material/info:")
        return

    job_ids = [job.id for job in current_jobs]
    selected_job_id = st.selectbox('Select Job ID', job_ids, index=0)

    st.write("\n")
    st.caption("**Evaluation Results**")

    selected_job = next((job for job in current_jobs if job.id == selected_job_id), None)

    if selected_job:
        csv_file_path = os.path.join(selected_job.evaluation_dir, 'result_evaluation.csv')

        if os.path.exists(csv_file_path):
            df = pd.read_csv(csv_file_path)
            st.data_editor(df, hide_index=True)
        else:
            st.info("Evaluation report not available yet. Please wait for MLflow run to complete.", icon=':material/info:')
    else:
        st.error("Selected job not found.")

# Main Application


display_page_header()


tab1, tab2 = st.tabs(["**MLflow Jobs List**", "**View MLflow Runs**"])

with tab1:
    display_jobs_list()

with tab2:
    display_mlflow_runs()
