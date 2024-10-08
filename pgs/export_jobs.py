import streamlit as st
import pandas as pd
import os
import requests
from google.protobuf.json_format import MessageToDict
from pgs.streamlit_utils import get_fine_tuning_studio_client
from ft.utils import format_status_with_icon
from ft.consts import IconPaths, DIVIDER_COLOR

fts = get_fine_tuning_studio_client()


def display_page_header():
    with st.container(border=True):
        col1, col2 = st.columns([1, 17])
        with col1:
            col1.image(IconPaths.CML.VIEW_EXPORT_JOBS)
        with col2:
            col2.subheader('View CML Model EXPORT jobs', divider=DIVIDER_COLOR)
            st.caption("Check the progress of CML Model EXPORT jobs")


def fetch_job_data():
    try:
        current_jobs = fts.get_export_jobs()
        models = fts.get_models()
        adapters = fts.get_adapters()

        model_dict = {model.id: model.name for model in models}
        adapter_dict = {adapter.id: adapter.name for adapter in adapters}

        return current_jobs, model_dict, adapter_dict
    except Exception as e:
        st.error(f"Error fetching job data: {e}")
        return [], {}, {}


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
    current_jobs, model_dict, adapter_dict = fetch_job_data()

    if not current_jobs:
        st.info("No CML export jobs triggered.", icon=":material/info:")
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
    columns_we_care_about = [
        'id',
        'model_name'
        'html_url',
        'latest',
        'base_model_name',
        'adapter_name',
        'created_at'
    ]

    for column in columns_we_care_about:
        if column not in display_df.columns:
            display_df[column] = 'Unknown'

    display_df = display_df[columns_we_care_about]

    display_df.rename(columns={
        'id': 'Job ID',
        'model_name': "CML Model Name",
        'base_model_name': 'Base Model Name',
        'adapter_name': 'Adapter Name',
        'latest': 'Status'
    }, inplace=True)

    display_df['Status'] = display_df['Status'].apply(
        lambda x: x['status'] if isinstance(x, dict) and 'status' in x else 'Unknown')
    display_df['status_with_icon'] = display_df['Status'].apply(format_status_with_icon)
    # Converting the 'created_at' column from a string in the format '2024-08-27T10:45:38.900Z' to a datetime object
    display_df['created_at'] = pd.to_datetime(display_df['created_at'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    display_df = display_df.sort_values(by='created_at', ascending=False)

    # Data editor for job table
    edited_df = st.data_editor(
        display_df[["Job ID", "status_with_icon", "created_at",
                    "html_url", "CML Model Name","Base Model Name", "Adapter Name"]],
        column_config={
            "Job ID": st.column_config.TextColumn("Job ID"),
            "status_with_icon": st.column_config.TextColumn(
                "Status",
                help="Job status as text with icon",
            ),
            "html_url": st.column_config.LinkColumn(
                "CML Job Link", display_text="Open CML Job"
            ),
            "CML Model Name": st.column_config.TextColumn("CML Model Name"),
            "Base Model Name": st.column_config.TextColumn("Base Model Name"),
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


display_page_header()

display_jobs_list()
