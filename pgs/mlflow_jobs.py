import streamlit as st
from ft.state import get_state
from ft.app import get_app
from st_aggrid import AgGrid
from st_aggrid import JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import pandas as pd
import os
import requests
import json

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
current_jobs = get_state().mlflow
models = get_state().models
adapters = get_state().adapters
datasets = get_app().datasets.list_datasets()

# Create dictionaries for ID to name mapping
model_dict = {model.id: model.name for model in models}
adapter_dict = {adapter.id: adapter.name for adapter in adapters}
dataset_dict = {dataset.id: dataset.name for dataset in datasets}

# Check if there are any current jobs
if not current_jobs:
    st.info("No MLflow jobs triggered.", icon=":material/info:")
else:
    col1, emptyCol, col2 = st.columns([32, 1, 30])
    with col1:
        st.subheader("MLflow Jobs List", divider='red')
        # Convert jobs to DataFrame
        try:
            jobs_df = pd.DataFrame([res.model_dump() for res in current_jobs])
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
                            right_on='public_identifier')

                        # Replace IDs with names using the dictionaries
                        display_df['adapter_name'] = display_df['adapter_id'].map(adapter_dict)
                        display_df['base_model_name'] = display_df['base_model_id'].map(model_dict)
                        display_df['dataset_name'] = display_df['dataset_id'].map(dataset_dict)

                        # Filter for only columns we care about
                        display_df = display_df[['job_id', 'html_url', 'latest',
                                                 'adapter_name', 'base_model_name', 'dataset_name']]

                        # Rename columns
                        display_df.rename(columns={
                            'job_id': 'Job ID',
                            'adapter_name': 'Adapter Name',
                            'base_model_name': 'Model Name',
                            'dataset_name': 'Dataset Name',
                            'latest': 'Status'
                        }, inplace=True)

                        # Apply status color renderer
                        display_df['Status'] = display_df['Status'].apply(lambda x: x['status'])

                        # Build Grid Options
                        gd = GridOptionsBuilder.from_dataframe(display_df)

                        # Renderer for links
                        cell_url_renderer = JsCode("""
                            class UrlCellRenderer {
                                init(params) {
                                    this.eGui = document.createElement('a');
                                    this.eGui.innerText = 'CML Job';
                                    this.eGui.setAttribute('href', params.value);
                                    this.eGui.setAttribute('style', "text-decoration:underline;");
                                    this.eGui.setAttribute('target', "_blank");
                                }
                                getGui() {
                                    return this.eGui;
                                }
                            }
                        """)
                        gd.configure_column("html_url", headerName="CML Job Link",
                                            cellRenderer=cell_url_renderer, width=300)

                        # Renderer for status colors
                        cell_status_renderer = JsCode("""
                            function(params) {
                                if (params.value == 'succeeded') {
                                    return {
                                        'color': 'white',
                                        'backgroundColor': 'green'
                                    }
                                } else if (params.value == 'running') {
                                    return {
                                        'color': 'white',
                                        'backgroundColor': 'blue'
                                    }
                                } else if (params.value == 'scheduling') {
                                    return {
                                        'color': 'black',
                                        'backgroundColor': 'grey'
                                    }
                                } else {
                                    return {
                                        'color': 'black',
                                        'backgroundColor': 'red'
                                    }
                                }
                            };
                        """)

                        gd.configure_column("Status", cellStyle=cell_status_renderer)

                        # Build all grid options
                        gridoptions = gd.build()

                        # Display the grid with the merged and filtered dataframe
                        AgGrid(
                            display_df,
                            gridOptions=gridoptions,
                            enable_enterprise_modules=False,
                            allow_unsafe_jscode=True,
                            height=540,
                            theme='alpine')

    with col2:
        st.subheader("View MLflow Job", divider='red')
        # Extract cml_job_ids
        job_ids = [job.job_id for job in current_jobs]

        # Select a cml_job_id from the list
        selected_job_id = st.selectbox('Select Job ID', job_ids, index=0)
        st.write("\n")
        st.caption("**Evaluation Results**")

        # Find the job corresponding to the selected job_id
        selected_job = next((job for job in current_jobs if job.job_id == selected_job_id), None)

        if selected_job:
            # Get the evaluation CSV file path for the selected job
            csv_file_path = os.path.join(selected_job.evaluation_dir, 'result_evaluation.csv')

            # Read the CSV file
            if os.path.exists(csv_file_path):
                df = pd.read_csv(csv_file_path)
                # Display the dataframe in Streamlit
                st.dataframe(df)
            else:
                st.info("Evaluation file not found.")
        else:
            st.error("Selected job not found.")
