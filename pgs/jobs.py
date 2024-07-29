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

# Function to read and return the trainer_state.json file


def get_trainer_json_data(job_id):
    """
    Search for the trainer_state.json file in outputs/{job_id} folder recursively
    and return its content as a dictionary.
    """
    base_path = os.path.join('outputs', job_id)
    file_path = None

    for root, dirs, files in os.walk(base_path):
        if 'trainer_state.json' in files:
            file_path = os.path.join(root, 'trainer_state.json')
            break

    if file_path:
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                return data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            return {}
    else:
        return {}


# Container for the layout
with st.container(border=True):
    col1, col2 = st.columns([1, 17])
    with col1:
        col1.image("./resources/images/subscriptions_24dp_EA3323_FILL0_wght400_GRAD0_opsz40.png")
    with col2:
        col2.subheader('Monitor Training Jobs')
        st.caption("Monitor your fine-tuning jobs, track progress, and ensure optimal performance throughout the training process.")

st.write("\n\n")

# Fetch current jobs from state
current_jobs = get_state().jobs
models = get_state().models
adapters = get_state().adapters
datasets = get_app().datasets.list_datasets()
prompts = get_state().prompts

# Create dictionaries for ID to name mapping
model_dict = {model.id: model.name for model in models}
adapter_dict = {adapter.id: adapter.name for adapter in adapters}
dataset_dict = {dataset.id: dataset.name for dataset in datasets}
prompt_dict = {prompt.id: prompt.name for prompt in prompts}

# Check if there are any current jobs
if not current_jobs:
    st.info("No fine-tuning jobs triggered.", icon=":material/info:")
else:
    col1, emptyCol, col2 = st.columns([30, 1, 22])
    with col1:
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
                        display_df['prompt_name'] = display_df['prompt_id'].map(prompt_dict)

                        # Filter for only columns we care about
                        display_df = display_df[['job_id', 'html_url', 'latest',
                                                 'adapter_name', 'base_model_name', 'dataset_name', 'prompt_name']]

                        # Rename columns
                        display_df.rename(columns={
                            'job_id': 'Job ID',
                            'adapter_name': 'Adapter Name',
                            'base_model_name': 'Model Name',
                            'dataset_name': 'Dataset Name',
                            'prompt_name': 'Prompt Name',
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
                            height=525,
                            theme='alpine')

    with col2:
        # Extract cml_job_ids
        job_ids = [job.job_id for job in current_jobs]

        # Select a cml_job_id from the list
        selected_job_id = st.selectbox('Select Job ID', job_ids, index=0)
        st.write("\n")

        # Display the trainer.json file for the selected cml_job_id
        training_data = get_trainer_json_data(selected_job_id)

        if training_data:
            # Extract data for plotting
            log_history = training_data.get("log_history", [])
            if log_history:
                df = pd.DataFrame(log_history)
                # Plotting using st.line_chart
                st.line_chart(df, x='epoch', y=['loss'], use_container_width=True)
            else:
                st.info("No log history found in the trainer_state.json file.")

            with st.expander("Show Training Data"):
                st.json(training_data)
        else:
            st.info(f"Training metrics not found for Job: {selected_job_id}")
