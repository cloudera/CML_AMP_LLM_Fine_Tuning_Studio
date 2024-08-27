import streamlit as st
import pandas as pd
import os
import requests
import json
from ft.api import *
import plotly.graph_objects as go
from google.protobuf.json_format import MessageToDict
from pgs.streamlit_utils import get_fine_tuning_studio_client, get_cml_client
from cmlapi import models as cml_api_models
from ft.utils import format_status_with_icon

# Instantiate the client to the FTS gRPC app server.
fts = get_fine_tuning_studio_client()
cml = get_cml_client()


def display_page_header():
    with st.container(border=True):
        col1, col2 = st.columns([1, 17])
        with col1:
            col1.image("./resources/images/subscriptions_24dp_EA3323_FILL0_wght400_GRAD0_opsz40.png")
        with col2:
            col2.subheader('Monitor Training Jobs')
            st.caption(
                "Monitor your fine-tuning jobs, track progress, and ensure optimal performance throughout the training process.")


def get_trainer_json_data(checkpoint_dir):
    file_path = os.path.join(checkpoint_dir, 'trainer_state.json')
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                return data
        except json.JSONDecodeError as e:
            st.error(f"Failed to decode JSON: {e}")
            return {}
        except Exception as e:
            st.error(f"Error reading trainer_state.json: {e}")
            return {}
    else:
        st.warning(f"trainer_state.json not found in {checkpoint_dir}.")
        return {}


def list_checkpoints(finetuning_framework, out_dir, job_id):
    try:
        if finetuning_framework == FineTuningFrameworkType.AXOLOTL:
            base_path = os.path.join(out_dir, job_id)
        else:
            base_path = os.path.join(os.getcwd(), 'outputs', job_id)

        checkpoints = {}
        if os.path.exists(base_path):
            for d in os.listdir(base_path):
                if d.startswith('checkpoint-'):
                    checkpoint_path = os.path.join(base_path, d)
                    checkpoints[d] = checkpoint_path
        return checkpoints
    except Exception as e:
        return {}


def fetch_current_jobs_and_mappings():
    try:
        current_jobs = fts.get_fine_tuning_jobs()
        models = fts.get_models()
        adapters = fts.get_adapters()
        datasets = fts.get_datasets()
        prompts = fts.get_prompts()

        model_dict = {model.id: model.name for model in models}
        adapter_dict = {adapter.id: adapter.name for adapter in adapters}
        dataset_dict = {dataset.id: dataset.name for dataset in datasets}
        prompt_dict = {prompt.id: prompt.name for prompt in prompts}

        return current_jobs, model_dict, adapter_dict, dataset_dict, prompt_dict
    except Exception as e:
        st.error(f"Error fetching jobs and mappings: {e}")
        return [], {}, {}, {}, {}


def fetch_cml_experiments():
    try:
        cml_project = os.getenv("CDSW_PROJECT_ID")
        all_experiments = []
        page_token = None
        page_size = 10

        while True:
            kwargs = {'page_size': page_size}
            if page_token:
                kwargs['page_token'] = page_token

            response = cml.list_experiments(cml_project, **kwargs).to_dict()
            all_experiments.extend(response['experiments'])

            page_token = response.get('next_page_token')
            if not page_token:
                break

        if len(all_experiments) == 0:
            cml_experiments_df = pd.DataFrame(columns=cml_api_models.Experiment().to_dict().keys())
        else:
            cml_experiments_df = pd.DataFrame(all_experiments)

        cml_experiments_df = cml_experiments_df[['id', 'name', 'artifact_location']].add_prefix('exp_')
        proj_url = os.getenv('CDSW_PROJECT_URL').replace("/api/v1/projects", "")
        cml_experiments_df['exp_id'] = cml_experiments_df['exp_id'].apply(lambda x: proj_url + "/cmlflow/" + x)
        return cml_experiments_df
    except Exception as e:
        st.error(f"Error fetching CML experiments: {e}")
        return pd.DataFrame()


def fetch_jobs_from_api():
    HOST = os.getenv('CDSW_PROJECT_URL')
    API_KEY = os.getenv('CDSW_API_KEY')

    if not HOST or not API_KEY:
        st.error("Environment variables for CDSW_PROJECT_URL or CDSW_API_KEY are missing.")
        return pd.DataFrame()

    url = "/".join([HOST, "jobs"])
    try:
        res = requests.get(url, headers={"Content-Type": "application/json"}, auth=(API_KEY, ""))
        res.raise_for_status()
    except requests.RequestException as e:
        st.error(f"Failed to fetch jobs from API: {e}")
        return pd.DataFrame()

    try:
        cml_jobs_list = res.json()
        return pd.DataFrame(cml_jobs_list)
    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"Error decoding API response: {e}")
        return pd.DataFrame()


@st.fragment
def display_jobs_list():
    current_jobs, model_dict, adapter_dict, dataset_dict, prompt_dict = fetch_current_jobs_and_mappings()
    cml_experiments_df = fetch_cml_experiments()
    if not current_jobs:
        st.info("No fine-tuning jobs triggered.", icon=":material/info:")
        return

    st.write("\n")

    _, col1 = st.columns([14, 2])

    with col1:
        if st.button("Reload", use_container_width=True, type="primary"):
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

    cml_jobs_list_df = fetch_jobs_from_api()
    if cml_jobs_list_df.empty or 'public_identifier' not in cml_jobs_list_df.columns:
        st.error("Column 'public_identifier' not found in cml_jobs_list_df")
        return

    display_df = pd.merge(
        jobs_df,
        cml_jobs_list_df,
        left_on='cml_job_id',
        right_on='public_identifier',
        how='left',
        suffixes=('', '_cml')
    )
    display_df = pd.merge(
        display_df,
        cml_experiments_df,
        left_on='id',
        right_on='exp_name',
        how='left',
        suffixes=('', '_cml')
    )

    display_df['base_model_name'] = display_df['base_model_id'].map(model_dict)
    display_df['dataset_name'] = display_df['dataset_id'].map(dataset_dict)
    if 'prompt_id' in display_df.columns:
        display_df['prompt_name'] = display_df['prompt_id'].apply(
            lambda pid: prompt_dict.get(pid, '') if pd.notnull(pid) else '')
    else:
        display_df['prompt_name'] = ''

    columns_we_care_about = [
        'id',
        'html_url',
        'latest',
        'base_model_name',
        'dataset_name',
        'prompt_name',
        'exp_id',
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
        'prompt_name': 'Prompt Name',
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
        display_df[["Job ID", "status_with_icon", "created_at", "html_url",
                    "exp_id", "Model Name", "Dataset Name", "Prompt Name"]],
        column_config={
            "Job ID": st.column_config.TextColumn("Job ID"),
            "status_with_icon": st.column_config.TextColumn(
                "Status",
                help="Job status as text with icon",
            ),
            "html_url": st.column_config.LinkColumn(
                "CML Job Link", display_text="Open CML Job"
            ),
            "exp_id": st.column_config.LinkColumn(
                "CML Exp Link", display_text="Open CML Exp"
            ),
            "Model Name": st.column_config.TextColumn("Model Name"),
            "Dataset Name": st.column_config.TextColumn("Dataset Name"),
            "Prompt Name": st.column_config.TextColumn("Prompt Name"),
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
    #                     response = fts.RemoveFineTuningJob(RemoveFineTuningJobRequest(
    #                         id=job_id
    #                     ))
    #                     st.toast(f"Job {job_id} deleted successfully.")
    #                 except Exception as e:
    #                     st.error(f"Error deleting job {job_id}: {str(e)}")

    #             # After all deletions, reload the specific component or data
    #             st.rerun(scope="fragment")
    #         else:
    #             st.warning("No jobs selected for deletion.")

    st.info(
        """
        Adapters generated from the fine-tuning job are available in the **View Base Models** page.
        Each adapter is associated with the specific base model it was trained on.
        """,
        icon=":material/info:"
    )


def display_training_metrics():
    current_jobs, model_dict, adapter_dict, dataset_dict, prompt_dict = fetch_current_jobs_and_mappings()

    if not current_jobs:
        st.info("No fine-tuning jobs triggered.", icon=":material/info:")
        return

    col1, _, col2 = st.columns([70, 1, 30])
    subcol1, subcol2 = col1.columns([6, 4])
    job_ids = [job.id for job in current_jobs]

    selected_job_id = subcol1.selectbox('Select Job ID', job_ids, index=0)

    selected_job = next((job for job in current_jobs if job.id == selected_job_id), None)
    if selected_job:
        finetuning_framework = selected_job.framework_type
        out_dir = selected_job.out_dir
    else:
        col1.error("Selected job not found.")
        return

    checkpoints = list_checkpoints(finetuning_framework, out_dir, selected_job_id)
    if not checkpoints:
        subcol2.selectbox('Select Checkpoint', [], index=0)
        col1.info(
            f"No checkpoints found for Job: **{selected_job_id}**. Please wait for job to save a checkpoint.",
            icon=":material/info:")

    else:
        checkpoint_names = list(checkpoints.keys())
        selected_checkpoint_name = subcol2.selectbox('Select Checkpoint', checkpoint_names, index=0)
        checkpoint_dir = checkpoints[selected_checkpoint_name]
        trainer_json_path = os.path.join(checkpoint_dir, 'trainer_state.json')

        try:
            with open(trainer_json_path, 'r') as file:
                training_data = json.load(file)
        except FileNotFoundError:
            col1.error("trainer_state.json not found.")
            training_data = None
        except json.JSONDecodeError as e:
            col1.error(f"Failed to decode JSON: {e}")
            training_data = None

        if not training_data:
            col1.info(f"Training metrics not found for Checkpoint: {selected_checkpoint_name}")
            return

        log_history = training_data.get("log_history", [])
        df = pd.DataFrame(log_history)

        if 'eval_loss' not in df.columns:
            df['eval_loss'] = pd.NA

        df = df.where(pd.notnull(df), None)

        if df.empty:
            st.info("No log history found in the trainer_state.json file.")
            return

        loss_df = df[['epoch', 'loss']].dropna()
        eval_loss_df = df[['epoch', 'eval_loss']].dropna()

        fig_loss = go.Figure()
        fig_eval_loss = go.Figure()

        fig_loss.add_trace(go.Scatter(
            x=loss_df['epoch'],
            y=loss_df['loss'],
            mode='lines+markers',
            name='Loss',
            line=dict(color='blue'),
            hovertemplate='Epoch: %{x}<br>Loss: %{y}<extra></extra>'
        ))

        fig_eval_loss.add_trace(go.Scatter(
            x=eval_loss_df['epoch'],
            y=eval_loss_df['eval_loss'],
            mode='lines+markers',
            name='Evaluation Loss',
            line=dict(color='red'),
            hovertemplate='Epoch: %{x}<br>Evaluation Loss: %{y}<extra></extra>'
        ))

        # Define the info icon text with HTML <br> for line breaks
        info_icon_text_loss = """
        Training loss represents how well the model is performing on the training data.
        """

        info_icon_text_eval_loss = """
        Evaluation loss, measures how well the model is performing on unseen data.
        """

        # Add an annotation for the info icon in the training loss plot
        fig_loss.update_layout(
            title='Training Loss',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            yaxis_type='log',
            height=600,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            annotations=[
                go.layout.Annotation(
                    text="ℹ️",
                    xref="paper", yref="paper",
                    x=1, y=1.15,  # Adjusted position to be within the chart area
                    showarrow=False,
                    font=dict(size=18),
                    hovertext=info_icon_text_loss,
                    hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="gray")
                )
            ]
        )

        # Add an annotation for the info icon in the evaluation loss plot
        fig_eval_loss.update_layout(
            title='Evaluation Loss',
            xaxis_title='Epoch',
            yaxis_title='Evaluation Loss',
            yaxis_type='log',
            height=600,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            annotations=[
                go.layout.Annotation(
                    text="ℹ️",
                    xref="paper", yref="paper",
                    x=1, y=1.15,  # Adjusted position to be within the chart area
                    showarrow=False,
                    font=dict(size=18),
                    hovertext=info_icon_text_eval_loss,
                    hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="gray")
                )
            ]
        )

        chart_col1, chart_col2 = col1.columns([1, 1])
        with chart_col1:
            st.plotly_chart(fig_loss, use_container_width=True)
        with chart_col2:
            st.plotly_chart(fig_eval_loss, use_container_width=True)

    col2.caption("**Job Metadata**")
    if selected_job:
        job_data = {
            "CPU": [selected_job.num_cpu],
            "GPU": [selected_job.num_gpu],
            "Memory": [selected_job.num_memory]
        }
        job_df = pd.DataFrame(job_data)
        col2.caption("Compute Configurations")
        col2.data_editor(job_df, use_container_width=True, hide_index=True)

        if selected_job.framework_type == FineTuningFrameworkType.LEGACY:
            try:
                train_config = fts.GetConfig(GetConfigRequest(id=selected_job.training_arguments_config_id))
                col2.caption("Training Configurations")
                col2.code(json.dumps(json.loads(train_config.config.config), indent=2), "json")
            except Exception as e:
                col2.error(f"Error fetching Training Config: {e}")

            try:
                lora_config = fts.GetConfig(GetConfigRequest(id=selected_job.lora_config_id))
                col2.caption("Lora Configurations")
                col2.code(json.dumps(json.loads(lora_config.config.config), indent=2), "json")
            except Exception as e:
                col2.error(f"Error fetching Lora Config: {e}")

            try:
                bnb_config = fts.GetConfig(GetConfigRequest(id=selected_job.adapter_bnb_config_id))
                col2.caption("BitsAndBytes Configurations")
                col2.code(json.dumps(json.loads(bnb_config.config.config), indent=2), "json")
            except Exception as e:
                col2.error(f"Error fetching BnB Config: {e}")

        elif selected_job.framework_type == FineTuningFrameworkType.AXOLOTL:
            try:
                axolotl_config = fts.GetConfig(GetConfigRequest(id=selected_job.axolotl_config_id))
                col2.caption("Axolotl Train Configurations")
                col2.code(axolotl_config.config.config, "yaml")
            except Exception as e:
                col2.error(f"Error fetching Axolotl Config: {e}")
        else:
            col2.error(f"Unsupported Finetuning framework used for: **{selected_job_id}**.", icon=":material/info:")
    else:
        col2.error("Selected job not found.")

# Main Application


display_page_header()

tab1, tab2 = st.tabs(["**View Jobs**", "**Job Details**"])

with tab1:
    display_jobs_list()

with tab2:
    display_training_metrics()
