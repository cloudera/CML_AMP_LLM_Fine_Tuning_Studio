import streamlit as st
import pandas as pd
import os
import requests
from google.protobuf.json_format import MessageToDict
from pgs.streamlit_utils import get_fine_tuning_studio_client, get_cml_client
from ft.utils import format_status_with_icon
from ft.consts import IconPaths, DIVIDER_COLOR, BASE_MODEL_ONLY_ADAPTER_ID, USER_DEFINED_IDENTIFIER, EVAL_INPUT_COLUMN, EVAL_OUTPUT_COLUM
from cmlapi import models as cml_api_models

# Instantiate the client to the FTS gRPC app server.
fts = get_fine_tuning_studio_client()
cml = get_cml_client()

# Note this is a duplicate code.from jobs.py. Problem with importing function from other page is that the page gets rendered too.
# total weird due to streamlit
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
            return pd.DataFrame(columns=cml_api_models.Experiment().to_dict().keys())[
                ['id', 'name', 'artifact_location']].add_prefix('exp_')

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

def display_page_header():
    with st.container(border=True):
        col1, col2 = st.columns([1, 17])
        with col1:
            col1.image(IconPaths.Experiments.VIEW_MLFLOW_RUNS)
        with col2:
            col2.subheader('View MLflow Runs', divider=DIVIDER_COLOR)
            st.caption("Examine the MLflow evaluation results for adapters trained on foundation models.")


def fetch_job_data():
    try:
        current_jobs = fts.get_evaluation_jobs()
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
        st.error(f"Error fetching job data: {e}")
        return [], {}, {}, {}, {}


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
    current_jobs, model_dict, adapter_dict, dataset_dict, prompt_dict = fetch_job_data()

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
    cml_experiments_df = fetch_cml_experiments()
    cml_experiments_df["parent_job_id"] = cml_experiments_df["exp_name"].apply(lambda x: x.split()[-1])
    display_df = pd.merge(
        jobs_df,
        cml_jobs_list_df,
        left_on='cml_job_id',
        right_on='public_identifier',
        suffixes=('', '_cml')
    )
    display_df = pd.merge(
        display_df,
        cml_experiments_df,
        how="left",
        on='parent_job_id')

    display_df['adapter_name'] = display_df['adapter_id'].map(adapter_dict)
    display_df['base_model_name'] = display_df['base_model_id'].map(model_dict)
    display_df['dataset_name'] = display_df['dataset_id'].map(dataset_dict)
    display_df['prompt_name'] = display_df['prompt_id'].map(prompt_dict)

    columns_we_care_about = [
        'parent_job_id',
        'id',
        'html_url',
        'latest',
        'base_model_name',
        'dataset_name',
        'adapter_name',
        'prompt_name',
        'created_at',
        'exp_id'
    ]

    for column in columns_we_care_about:
        if column not in display_df.columns:
            display_df[column] = 'Unknown'

    display_df = display_df[columns_we_care_about]

    display_df.rename(columns={
        'parent_job_id': 'Parent Job ID',
        'id': 'Job ID',
        'base_model_name': 'Model Name',
        'dataset_name': 'Dataset Name',
        'adapter_name': 'Adapter Name',
        'prompt_name': 'Prompt Name',
        'latest': 'Status',
        'exp_id': 'Experiment Url'
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
        display_df[["Parent Job ID", "Job ID", "status_with_icon", "created_at",
                    "html_url", "Experiment Url", "Model Name", "Adapter Name", "Dataset Name", "Prompt Name"]],
        column_config={
            "Parent Job ID": st.column_config.TextColumn("Parent Job ID"),
            "Job ID": st.column_config.TextColumn("Job ID"),
            "status_with_icon": st.column_config.TextColumn(
                "Status",
                help="Job status as text with icon",
            ),
            "html_url": st.column_config.LinkColumn(
                "CML Job Link", display_text="Open CML Job"
            ),
            "Experiment Url": st.column_config.LinkColumn(
                "Experiment Link", display_text="Open Experiment"
            ),
            "Model Name": st.column_config.TextColumn("Model Name"),
            "Dataset Name": st.column_config.TextColumn("Dataset Name"),
            "Adapter Name": st.column_config.TextColumn("Adapter Name"),
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
    current_jobs, model_dict, adapter_dict, dataset_dict, _ = fetch_job_data()
    if not current_jobs:
        st.info("No MLflow jobs triggered.", icon=":material/info:")
        return

    parent_job_ids = list(set([job.parent_job_id for job in current_jobs]))
    selected_parent_job_id = st.selectbox('Select Parent Job ID', parent_job_ids, index=0)

    st.write("\n")
    st.caption("**Evaluation Results**")
    selected_jobs = [job for job in current_jobs if job.parent_job_id == selected_parent_job_id]

    if selected_jobs is not None:
        st.write("\n")
        st.write(f"Dataset used : {dataset_dict[selected_jobs[0].dataset_id]}")
        all_aggreated_paths = []
        for selected_job in selected_jobs:
            aggregated_file_path = os.path.join(selected_job.evaluation_dir, "aggregregated_results.csv")
            csv_file_path = os.path.join(selected_job.evaluation_dir, 'result_evaluation.csv')
            all_aggreated_paths.append({"aggregated_csv": aggregated_file_path, "row_wise_csv": csv_file_path,
                                        "model_name": model_dict[selected_job.base_model_id],
                                        "adapter_name": adapter_dict[selected_job.adapter_id] if selected_job.adapter_id != BASE_MODEL_ONLY_ADAPTER_ID else ""})
        final_df = None
        for idx, all_aggreated_path in enumerate(all_aggreated_paths):
            evaluation_name = f"{all_aggreated_path['model_name']}"
            if all_aggreated_path['adapter_name']:
                evaluation_name += f" + {all_aggreated_path['adapter_name']}"
            if os.path.exists(all_aggreated_path['aggregated_csv']):
                df_ar = pd.read_csv(all_aggreated_path['aggregated_csv'])
                # st.data_editor(df_ar)
                df_ar.columns = ["metric", evaluation_name]
                if final_df is None:
                    final_df = df_ar
                else:
                    final_df = pd.merge(final_df, df_ar, on="metric")
        if final_df is not None:
            st.write("\n")
            st.caption("**Aggregated Results**")
            st.data_editor(final_df, hide_index=True)
            st.write("\n")
            st.caption("**Row Wise Results**")
        non_metric_columns = [EVAL_INPUT_COLUMN, EVAL_OUTPUT_COLUM]
        flag = False
        final_df = None
        for all_aggreated_path in all_aggreated_paths:
            evaluation_name = f"{all_aggreated_path['model_name']}"
            if all_aggreated_path['adapter_name']:
                evaluation_name += f" + {all_aggreated_path['adapter_name']}"
            csv_file_path = all_aggreated_path.get("row_wise_csv")
            if csv_file_path and os.path.exists(csv_file_path):
                df = pd.read_csv(csv_file_path)
                if not flag:
                    for col in list(df.columns):
                        if USER_DEFINED_IDENTIFIER in col:
                            non_metric_columns.append(col.split(USER_DEFINED_IDENTIFIER)[0])
                    flag = True
                col_map = {}
                for col in list(df.columns):
                    target_col = col.split(USER_DEFINED_IDENTIFIER)[0]
                    if target_col in non_metric_columns:
                        col_map[col] = target_col
                    else:
                        col_map[col] = col + "\n" + " " + evaluation_name
                df.rename(columns=col_map, inplace=True)
                if final_df is None:
                    final_df = df
                else:
                    final_df = pd.merge(final_df, df, on=non_metric_columns)
        st.data_editor(final_df, hide_index=True)

        # if os.path.exists(csv_file_path):
        #     st.write("\n")
        #     st.caption("**Row Wise Results**")
        #     df = pd.read_csv(csv_file_path)
        #     st.data_editor(df, hide_index=True)
        if final_df is None:
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
