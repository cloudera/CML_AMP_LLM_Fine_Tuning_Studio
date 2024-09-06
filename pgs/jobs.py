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
from ft.consts import IconPaths, DIVIDER_COLOR
import math

# Instantiate the client to the FTS gRPC app server.
fts = get_fine_tuning_studio_client()
cml = get_cml_client()


def display_page_header():
    with st.container():
        col1, col2 = st.columns([1, 17])
        with col1:
            col1.image(IconPaths.Experiments.MONITOR_TRAINING_JOBS)
        with col2:
            col2.subheader('Monitor Training Jobs', divider=DIVIDER_COLOR)
            st.caption(
                "Monitor your fine-tuning jobs, track progress, and ensure optimal performance throughout the training process.")


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
        st.error(f"Error fetching jobs and mappings: {e}", icon=":material/error:")
        return [], {}, {}, {}, {}


def fetch_cml_experiments():
    try:
        cml_project = os.getenv("CDSW_PROJECT_ID")
        if not cml_project:
            st.error("CDSW_PROJECT_ID environment variable is missing.", icon=":material/error:")
            return pd.DataFrame()

        all_experiments = []
        page_token = None
        page_size = 10

        while True:
            kwargs = {'page_size': page_size}
            if page_token:
                kwargs['page_token'] = page_token

            response = cml.list_experiments(cml_project, **kwargs).to_dict()
            all_experiments.extend(response.get('experiments', []))

            page_token = response.get('next_page_token')
            if not page_token:
                break

        if not all_experiments or len(all_experiments) == 0:
            return pd.DataFrame(columns=cml_api_models.Experiment().to_dict().keys())[['id', 'name', 'artifact_location']].add_prefix('exp_')

        cml_experiments_df = pd.DataFrame(all_experiments)

        cml_experiments_df = cml_experiments_df[['id', 'name', 'artifact_location']].add_prefix('exp_')
        proj_url = os.getenv('CDSW_PROJECT_URL', '').replace("/api/v1/projects", "")
        if not proj_url:
            st.error("CDSW_PROJECT_URL environment variable is missing or invalid.", icon=":material/error:")
            return pd.DataFrame()

        cml_experiments_df['exp_id'] = cml_experiments_df['exp_id'].apply(lambda x: proj_url + "/cmlflow/" + x)
        return cml_experiments_df
    except Exception as e:
        st.error(f"Error fetching CML experiments: {e}", icon=":material/error:")
        return pd.DataFrame()


def fetch_jobs_from_api():
    HOST = os.getenv('CDSW_PROJECT_URL')
    API_KEY = os.getenv('CDSW_API_KEY')

    if not HOST or not API_KEY:
        st.error("Environment variables for CDSW_PROJECT_URL or CDSW_API_KEY are missing.", icon=":material/error:")
        return pd.DataFrame()

    url = "/".join([HOST, "jobs"])
    try:
        res = requests.get(url, headers={"Content-Type": "application/json"}, auth=(API_KEY, ""))
        res.raise_for_status()
    except requests.RequestException as e:
        st.error(f"Failed to fetch jobs from API: {e}", icon=":material/error:")
        return pd.DataFrame()

    try:
        cml_jobs_list = res.json()
        return pd.DataFrame(cml_jobs_list)
    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"Error decoding API response: {e}", icon=":material/error:")
        return pd.DataFrame()


def fetch_job_status(job_id):
    HOST = os.getenv('CDSW_PROJECT_URL')
    API_KEY = os.getenv('CDSW_API_KEY')

    if not HOST or not API_KEY:
        st.error("Environment variables for CDSW_PROJECT_URL or CDSW_API_KEY are missing.", icon=":material/error:")
        return None

    url = f"{HOST}/jobs"
    try:
        res = requests.get(url, headers={"Content-Type": "application/json"}, auth=(API_KEY, ""))
        res.raise_for_status()
    except requests.RequestException as e:
        st.error(f"Failed to fetch job status from API: {e}", icon=":material/error:")
        return None

    try:
        jobs_list = res.json()
        if not isinstance(jobs_list, list):
            st.error("Unexpected API response format: expected a list of jobs.", icon=":material/error:")
            return None

        matching_job = next((job for job in jobs_list if job.get('public_identifier') == job_id), None)

        if not matching_job:
            st.warning(f"No job found with the ID: {job_id}", icon=":material/warning:")
            return None

        job_status = matching_job.get('latest', {}).get('status', 'Unknown')
        return job_status

    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"Error decoding API response: {e}", icon=":material/error:")
        return None


def fetch_experiment_metrics(exp_id):
    cml_project = os.getenv("CDSW_PROJECT_ID")

    if not cml_project:
        st.error("CDSW_PROJECT_ID environment variable is missing.", icon=":material/error:")
        return []

    if len(exp_id) != 19:
        st.error("Invalid experiment ID length.", icon=":material/error:")
        return []

    try:
        experiment_details = cml.get_experiment(project_id=cml_project, experiment_id=exp_id).to_dict()
    except Exception as e:
        st.error(f"Error fetching experiment details: {e}", icon=":material/error:")
        return []

    try:
        experiment_runs_response = cml.list_experiment_runs(project_id=cml_project, experiment_id=exp_id).to_dict()
        experiment_runs = experiment_runs_response.get('experiment_runs', [])

        if not experiment_runs:
            return []

        first_run = experiment_runs[0]
        run_id = first_run.get('id')

    except Exception as e:
        st.error(f"Error fetching experiment runs: {e}", icon=":material/error:")
        return []

    metrics = first_run.get('data', {}).get('metrics', [])
    if not metrics:
        st.info("No metrics found for the experiment.", icon=":material/info:")
        return []

    metric_data_list = []
    for metric in metrics:
        metric_key = metric.get('key')
        try:
            metric_data = cml.get_experiment_run_metrics(
                project_id=cml_project,
                experiment_id=exp_id,
                run_id=run_id,
                metric_key=metric_key
            ).to_dict()

            metric_data_list.append({'metric_key': metric_key, 'metric_data': metric_data['metrics']})
        except Exception as e:
            st.error(f"Failed to fetch metric {metric_key} for run {run_id}: {e}", icon=":material/error:")

    return metric_data_list


@st.fragment
def display_jobs_list():
    current_jobs, model_dict, adapter_dict, dataset_dict, prompt_dict = fetch_current_jobs_and_mappings()
    cml_experiments_df = fetch_cml_experiments()
    if not current_jobs:
        st.info("No fine-tuning jobs triggered.", icon=":material/info:")
        return

    st.write("\n")

    _, col1 = st.columns([18, 2])

    with col1:
        if st.button("Refresh", use_container_width=True, type="primary"):
            st.rerun(scope="fragment")

    try:
        jobs_df = pd.DataFrame([MessageToDict(res, preserving_proto_field_name=True) for res in current_jobs])
    except Exception as e:
        st.error(f"Error converting jobs to DataFrame: {e}", icon=":material/error:")
        jobs_df = pd.DataFrame()

    if 'cml_job_id' not in jobs_df.columns:
        st.error("Column 'cml_job_id' not found in jobs_df", icon=":material/error:")
        return

    cml_jobs_list_df = fetch_jobs_from_api()
    if cml_jobs_list_df.empty or 'public_identifier' not in cml_jobs_list_df.columns:
        st.error("Column 'public_identifier' not found in cml_jobs_list_df", icon=":material/error:")
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
    display_df['prompt_name'] = display_df['prompt_id'].apply(
        lambda pid: prompt_dict.get(pid, '') if pd.notnull(pid) else '')

    columns_we_care_about = [
        'id', 'html_url', 'latest', 'base_model_name', 'dataset_name', 'prompt_name',
        'exp_id', 'created_at', 'adapter_name'
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
        'latest': 'Status',
        'adapter_name': 'Adapter Name'
    }, inplace=True)

    display_df['Status'] = display_df['Status'].apply(
        lambda x: x['status'] if isinstance(x, dict) and 'status' in x else 'Unknown')
    display_df['status_with_icon'] = display_df['Status'].apply(format_status_with_icon)

    display_df['created_at'] = pd.to_datetime(display_df['created_at'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    display_df = display_df.sort_values(by='created_at', ascending=False)

    edited_df = st.data_editor(
        display_df[["Job ID", "status_with_icon", "Adapter Name", "html_url",
                    "exp_id", "created_at", "Model Name", "Dataset Name", "Prompt Name"]],
        column_config={
            "Job ID": st.column_config.TextColumn("Job ID"),
            "status_with_icon": st.column_config.TextColumn(
                "Status",
                help="Job status as text with icon",
            ),
            "Adapter Name": st.column_config.TextColumn("Adapter Name"),
            "html_url": st.column_config.LinkColumn(
                "CML Job Link", display_text="Open CML Job"
            ),
            "exp_id": st.column_config.LinkColumn(
                "CML Exp Link", display_text="Open CML Exp"
            ),
            "Model Name": st.column_config.TextColumn("Model Name"),
            "Dataset Name": st.column_config.TextColumn("Dataset Name"),
            "Prompt Name": st.column_config.TextColumn("Prompt Name"),
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

    st.info(
        """
        Adapters generated from the fine-tuning job are available in the **View Base Models** page.
        Each adapter is associated with the specific base model it was trained on.
        """,
        icon=":material/info:"
    )


@st.fragment
def display_training_metrics():
    current_jobs, model_dict, adapter_dict, dataset_dict, prompt_dict = fetch_current_jobs_and_mappings()
    cml_experiments_df = fetch_cml_experiments()
    cml_jobs_df = fetch_jobs_from_api()

    if not current_jobs:
        st.info("No fine-tuning jobs triggered.", icon=":material/info:")
        return

    col1, _, col2 = st.columns([70, 1, 30])

    job_details = [
        f"{job.id} [ Adapter = {job.adapter_name} ]"
        for job in current_jobs
    ]

    selected_job_detail = col1.selectbox('Select Job ID', job_details)

    with col2:
        _, subcol1 = st.columns([2, 1])
        with subcol1:
            st.write("\n")
            if st.button("Refresh", use_container_width=True, type="primary", key="refresh_job_details"):
                st.rerun(scope="fragment")

    if selected_job_detail:
        selected_job_id = selected_job_detail.split(" ")[0]
    else:
        st.info("Please select a job.", icon=":material/info:")
        return

    selected_job = next((job for job in current_jobs if job.id == selected_job_id), None)
    if not selected_job:
        st.info("Selected job could not be found.", icon=":material/info:")
        return

    progress = 0

    if selected_job:
        with col1:
            job_status = fetch_job_status(selected_job.cml_job_id)
            if job_status is None:
                job_status = "Unknown"

            if job_status.lower() in ['failed', 'stopped', 'unknown']:
                st.error(
                    f"The selected job is in '**{job_status}**' status. Please go to the CML job page to check the logs and failure reasons.",
                    icon=":material/error:")
            else:
                matching_experiments = cml_experiments_df[cml_experiments_df['exp_name'] == selected_job_id]

                if not matching_experiments.empty:

                    experiment_id = matching_experiments.iloc[0]['exp_id'].split("/")[-1]

                    metric_data_list = fetch_experiment_metrics(experiment_id)

                    if metric_data_list:

                        loss_data = []
                        eval_loss_data = []
                        epochs_data = []

                        for metric in metric_data_list:
                            if metric['metric_key'] == 'loss':
                                loss_data = metric['metric_data']
                            elif metric['metric_key'] == 'eval_loss':
                                eval_loss_data = metric['metric_data']
                            elif metric['metric_key'] == 'epoch':
                                epochs_data = metric['metric_data']

                        if epochs_data:
                            progress = math.ceil((epochs_data[-1]['value'] / selected_job.num_epochs) * 100)

                        fig_loss = go.Figure()
                        fig_eval_loss = go.Figure()

                        if loss_data:
                            fig_loss.add_trace(go.Scatter(
                                x=[data['timestamp'] for data in loss_data if 'timestamp' in data and 'value' in data],
                                y=[data['value'] for data in loss_data if 'timestamp' in data and 'value' in data],
                                mode='lines+markers',
                                name='Loss',
                                line=dict(color='blue'),
                                hovertemplate='Timestamp: %{x}<br>Loss: %{y}<extra></extra>'
                            ))

                        if eval_loss_data:
                            fig_eval_loss.add_trace(go.Scatter(
                                x=[data['timestamp'] for data in eval_loss_data if 'timestamp' in data and 'value' in data],
                                y=[data['value'] for data in eval_loss_data if 'timestamp' in data and 'value' in data],
                                mode='lines+markers',
                                name='Evaluation Loss',
                                line=dict(color='red'),
                                hovertemplate='Timestamp: %{x}<br>Evaluation Loss: %{y}<extra></extra>'
                            ))

                        info_icon_text_loss = """
                        Training loss represents how well the model is performing on the training data.
                        """

                        info_icon_text_eval_loss = """
                        Evaluation loss measures how well the model is performing on unseen data.
                        """

                        fig_loss.update_layout(
                            title='Training Loss',
                            xaxis_title='Time',
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
                                    x=1, y=1.15,
                                    showarrow=False,
                                    font=dict(size=18),
                                    hovertext=info_icon_text_loss,
                                    hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="gray")
                                )
                            ]
                        )

                        fig_eval_loss.update_layout(
                            title='Evaluation Loss',
                            xaxis_title='Time',
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
                                    x=1, y=1.15,
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
                    else:
                        col1.info("No metrics data found to display.", icon=":material/info:")

                else:
                    col1.info(
                        f"No metrics found for Job ID: {selected_job_id}. Please wait for job to start training base model.",
                        icon=":material/info:")

        with col2:
            col2.markdown("\n")
            col2.caption("**Job Metadata**")

            progress_data = {
                "Progress (%)": [progress]
            }
            progress_df = pd.DataFrame(progress_data)

            progress_column_config = {
                "Progress (%)": st.column_config.ProgressColumn(
                    label="Progress (%)",
                    help="Training progress as a percentage",
                    format="%f%%",
                    min_value=0,
                    max_value=100
                )
            }

            col2.data_editor(
                progress_df,
                use_container_width=True,
                hide_index=True,
                column_config=progress_column_config)

            job_metadata = {
                "Attribute": ["Adapter", "Prompt", "Base Model"],
                "Value": [
                    selected_job.adapter_name,
                    prompt_dict.get(selected_job.prompt_id, "Unknown"),
                    model_dict.get(selected_job.base_model_id, "Unknown")
                ]
            }
            job_metadata_df = pd.DataFrame(job_metadata)
            col2.data_editor(job_metadata_df, use_container_width=True, hide_index=True)

            job_data = {
                "CPU": [selected_job.num_cpu],
                "GPU": [selected_job.num_gpu],
                "Memory": [selected_job.num_memory]
            }
            job_df = pd.DataFrame(job_data)
            col2.caption("Compute Configurations")
            col2.data_editor(job_df, use_container_width=True, hide_index=True)

            if selected_job.framework_type == FineTuningFrameworkType.LEGACY:
                with col2.expander("Training Configurations", expanded=False):
                    try:
                        train_config = fts.GetConfig(GetConfigRequest(id=selected_job.training_arguments_config_id))
                        st.code(json.dumps(json.loads(train_config.config.config), indent=2), "json")
                    except Exception as e:
                        st.error(f"Error fetching Training Config: {e}", icon=":material/error:")

                with col2.expander("Lora Configurations", expanded=False):
                    try:
                        lora_config = fts.GetConfig(GetConfigRequest(id=selected_job.lora_config_id))
                        st.code(json.dumps(json.loads(lora_config.config.config), indent=2), "json")
                    except Exception as e:
                        st.error(f"Error fetching Lora Config: {e}", icon=":material/error:")

                with col2.expander("BitsAndBytes Configurations", expanded=False):
                    try:
                        bnb_config = fts.GetConfig(GetConfigRequest(id=selected_job.adapter_bnb_config_id))
                        st.code(json.dumps(json.loads(bnb_config.config.config), indent=2), "json")
                    except Exception as e:
                        st.error(f"Error fetching BnB Config: {e}", icon=":material/error:")

            elif selected_job.framework_type == FineTuningFrameworkType.AXOLOTL:
                with col2.expander("Axolotl Train Configurations", expanded=False):
                    try:
                        axolotl_config = fts.GetConfig(GetConfigRequest(id=selected_job.axolotl_config_id))
                        st.code(axolotl_config.config.config, "yaml")
                    except Exception as e:
                        st.error(f"Error fetching Axolotl Config: {e}", icon=":material/error:")
            else:
                col2.error(
                    f"Unsupported Finetuning framework used for: **{selected_job_id}**.",
                    icon=":material/error:")

    else:
        col2.error("Selected job not found.", icon=":material/error:")


# Main Application
display_page_header()

tab1, tab2 = st.tabs(["**View Jobs**", "**Job Details**"])

with tab1:
    display_jobs_list()

with tab2:
    display_training_metrics()
