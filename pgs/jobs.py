import streamlit as st
from ft.state import get_state
from ft.app import get_app
import pandas as pd
import os
import requests
import json
import altair as alt
import cmlapi
import plotly.graph_objects as go
from typing import List

# Function to read and return the trainer_state.json file
def get_trainer_json_data(checkpoint_dir):
    """
    Search for the trainer_state.json file in the specified checkpoint_dir folder
    and return its content as a dictionary.
    """
    file_path = os.path.join(checkpoint_dir, 'trainer_state.json')

    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                return data
        except Exception as e:
            return {}
    else:
        return {}


def list_checkpoints(job_id):
    """
    List all checkpoint folders for a given job_id in outputs/{job_id}.
    """
    try:
        base_path = os.path.join('outputs', job_id)
        checkpoints = [d for d in os.listdir(base_path) if d.startswith('checkpoint-')]
        return checkpoints
    except Exception as e:
        return []


def generate_experiments_dataframe(experiments: List) -> pd.DataFrame:
    """
    Generate a render-able dataframe of experiments based on available
    experiments in a CML workspace.
    """
    
    cml_experiments_df = pd.DataFrame(experiments)
    
    if experiments:
        cml_experiments_df = cml_experiments_df[['id', 'name', 'artifact_location']]
        cml_experiments_df = cml_experiments_df.add_prefix('exp_')

        # Resolve real URL for each experiment
        proj_url = os.getenv('CDSW_PROJECT_URL').replace("/api/v1/projects", "")
        cml_experiments_df['exp_id'] = cml_experiments_df['exp_id'].apply(lambda x: proj_url + "/cmlflow/" + x)
    
    return cml_experiments_df


# Initialize CML API v2 Client
cml_api_client = cmlapi.default_client()
cml_project = os.getenv("CDSW_PROJECT_ID")

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

# Fetch current experiments tracked in CML
all_experiments = []
page_token = None
page_size = 10  # Set your desired page size here

while True:
    kwargs = {'page_size': page_size}
    if page_token:
        kwargs['page_token'] = page_token
    
    response = cml_api_client.list_experiments(
        cml_project,
        **kwargs
    ).to_dict()
    
    all_experiments.extend(response['experiments'])
    
    page_token = response.get('next_page_token')
    if not page_token:
        break

# Container for the layout
with st.container(border=True):
    col1, col2 = st.columns([1, 17])
    with col1:
        col1.image("./resources/images/subscriptions_24dp_EA3323_FILL0_wght400_GRAD0_opsz40.png")
    with col2:
        col2.subheader('Monitor Training Jobs')
        st.caption("Monitor your fine-tuning jobs, track progress, and ensure optimal performance throughout the training process.")

st.write("\n")

# Streamlit tabs
tab1, tab2 = st.tabs(["**Jobs List**", "**View Jobs**"])

with tab1:
    # Check if there are any current jobs
    if not current_jobs:
        st.info("No fine-tuning jobs triggered.", icon=":material/info:")
    else:
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
                        
                        # Merge the DataFrame for experiments
                        display_df = pd.merge(
                            display_df,
                            generate_experiments_dataframe(all_experiments),
                            left_on='job_id',
                            right_on='exp_name')

                        # Replace IDs with names using the dictionaries
                        display_df['adapter_name'] = display_df['adapter_id'].map(adapter_dict)
                        display_df['base_model_name'] = display_df['base_model_id'].map(model_dict)
                        display_df['dataset_name'] = display_df['dataset_id'].map(dataset_dict)
                        display_df['prompt_name'] = display_df['prompt_id'].map(prompt_dict)

                        # Filter for only columns we care about
                        display_df = display_df[['job_id', 'html_url', 'latest',
                                                    'adapter_name', 'base_model_name', 'dataset_name',
                                                    'prompt_name', 'exp_id']]
                        
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

                        status_mapping = {
                            "succeeded": 100,
                            "running": 30,
                            "scheduling": 1
                        }
                        display_df['status'] = display_df['Status'].apply(
                            lambda x: status_mapping.get(x, 0) if pd.notnull(x) else 0)

                        # Display the grid with the merged and filtered dataframe
                        st.data_editor(
                            display_df[["Job ID", "status", "html_url", "exp_id", "Adapter Name", "Model Name", "Dataset Name", "Prompt Name"]],
                            column_config={
                                "Job ID": st.column_config.TextColumn("Job ID"),
                                "status": st.column_config.ProgressColumn(
                                    "Status",
                                    help="Job status as progress",
                                    format="%.0f%%",
                                    min_value=0,
                                    max_value=100,
                                ),
                                "html_url": st.column_config.LinkColumn(
                                    "CML Job Link", display_text="Open CML Job"
                                ),
                                "exp_id": st.column_config.LinkColumn(
                                    "CML Exp Link", display_text="Open CML Exp"
                                ),
                                "Adapter Name": st.column_config.TextColumn("Adapter Name"),
                                "Model Name": st.column_config.TextColumn("Model Name"),
                                "Dataset Name": st.column_config.TextColumn("Dataset Name"),
                                "Prompt Name": st.column_config.TextColumn("Prompt Name")
                            },
                            height=500,
                            hide_index=True,
                            use_container_width=True
                        )

                        st.info(
                            """
                            Adapters generated from the fine-tuning job are available in the **View Base Models** page.
                            Each adapter is associated with the specific base model it was trained on.
                            """,
                            icon=":material/info:"
                        )

with tab2:
    # Check if there are any current jobs
    if not current_jobs:
        st.info("No fine-tuning jobs triggered.", icon=":material/info:")
    else:
        col1, col2, col3 = st.columns([2, 1, 3])
        job_ids = [job.job_id for job in current_jobs]

        # Select a cml_job_id from the list
        selected_job_id = col1.selectbox('Select Job ID', job_ids, index=0)

        # Get the list of checkpoints for the selected job and sort them numerically
        checkpoints = sorted(list_checkpoints(selected_job_id), key=lambda x: int(x.split('-')[-1]))

        if checkpoints:
            # Select a checkpoint from the list
            selected_checkpoint = col2.selectbox('Select Checkpoint', checkpoints, index=0)

            # Display the trainer.json file for the selected checkpoint
            trainer_json_path = os.path.join('outputs', selected_job_id, selected_checkpoint, 'trainer_state.json')
            try:
                with open(trainer_json_path, 'r') as file:
                    training_data = json.load(file)
            except FileNotFoundError:
                st.error("File not found.")
                training_data = None

            if training_data:
                log_history = training_data.get("log_history", [])
                df = pd.DataFrame(log_history)

                # Ensure the 'eval_loss' column exists without forward-filling
                if 'eval_loss' not in df.columns:
                    df['eval_loss'] = pd.NA  # Ensure the column exists

                # Replace pd.NA with None
                df = df.where(pd.notnull(df), None)

                if not df.empty:
                    # Separate dataframes for loss and eval_loss
                    loss_df = df[['epoch', 'loss']].dropna()
                    eval_loss_df = df[['epoch', 'eval_loss']].dropna()

                    fig_loss = go.Figure()
                    fig_eval_loss = go.Figure()

                    # Add Loss trace
                    fig_loss.add_trace(go.Scatter(
                        x=loss_df['epoch'],
                        y=loss_df['loss'],
                        mode='lines+markers',
                        name='Loss',
                        line=dict(color='blue'),
                        hovertemplate='Epoch: %{x}<br>Loss: %{y}<extra></extra>'
                    ))

                    # Add Evaluation Loss trace
                    fig_eval_loss.add_trace(go.Scatter(
                        x=eval_loss_df['epoch'],
                        y=eval_loss_df['eval_loss'],
                        mode='lines+markers',
                        name='Evaluation Loss',
                        line=dict(color='red'),
                        hovertemplate='Epoch: %{x}<br>Evaluation Loss: %{y}<extra></extra>'
                    ))

                    # Update layout for both figures
                    fig_loss.update_layout(
                        title='Training Loss',
                        xaxis_title='Epoch',
                        yaxis_title='Loss',
                        yaxis_type='log',
                        height=500,
                        legend=dict(
                            orientation='h',
                            yanchor='bottom',
                            y=1.02,
                            xanchor='right',
                            x=1
                        )
                    )

                    fig_eval_loss.update_layout(
                        title='Evaluation Loss',
                        xaxis_title='Epoch',
                        yaxis_title='Evaluation Loss',
                        yaxis_type='log',
                        height=500,
                        legend=dict(
                            orientation='h',
                            yanchor='bottom',
                            y=1.02,
                            xanchor='right',
                            x=1
                        )
                    )

                    # Create separate columns for charts and training data
                    col11, col12 = st.columns([7,3])
                    with col11:
                        chart_col1, chart_col2 = st.columns([5, 5])
                        with chart_col1:
                            st.plotly_chart(fig_loss, use_container_width=True)
                        with chart_col2:
                            st.plotly_chart(fig_eval_loss, use_container_width=True)

                    with col12:
                        st.json(training_data)

                    col21, col22 = st.columns([7,3])
                    with col11:
                        st.caption("Job Metadata")

                        # Find the selected job metadata
                        selected_job = next((job for job in current_jobs if job.job_id == selected_job_id), None)

                        if selected_job:                        
                            # Prepare data for display in Streamlit data editor
                            job_data = {
                                "Epochs": [selected_job.num_epochs],
                                "Learning Rate": [selected_job.learning_rate],
                                "CPU": [selected_job.worker_props.num_cpu],
                                "GPU": [selected_job.worker_props.num_gpu],
                                "Memory": [selected_job.worker_props.num_memory]
                            }

                            # Convert data to DataFrame for display
                            job_df = pd.DataFrame(job_data)
                            
                            # Display data editor
                            st.data_editor(job_df, use_container_width=True, hide_index=True)
                        else:
                            st.error("Selected job not found.")

                    with col12:
                        st.json(training_data)
                else:
                    st.info("No log history found in the trainer_state.json file.")
            else:
                st.info(f"Training metrics not found for Checkpoint: {selected_checkpoint}")
        else:
            st.info(f"No checkpoints found for Job: **{selected_job_id}**. Please wait for job to complete", icon=":material/info:")
