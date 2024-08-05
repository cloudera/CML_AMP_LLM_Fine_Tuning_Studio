import streamlit as st
import ft.app
from ft.app import FineTuningAppProps, FineTuningApp
from ft.state import get_state_location
from ft.managers import (
    DatasetsManagerSimple,
    ModelsManagerSimple,
    FineTuningJobsManagerSimple,
    MLflowEvaluationJobsManagerSimple
)
import streamlit.components.v1 as components
from ft.utils import get_env_variable

# Module for custom CSS


def apply_custom_css():
    css = '''
    <style>
        h3 {
            font-size: 1.1rem;
        }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 0.9rem;
        }
        [data-testid="stHeader"] {
            color: #4CAF50;
            background-color: #f0f0f0;
        }
        [data-testid="stSidebarContent"] {
            background: #16262c;
            color: white;
        }
        [data-testid="stSidebarContent"] * {
            color: white;
        }
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)

# Module for initialization of FineTuningApp


def initialize_ft_app():
    ft.app.INSTANCE = FineTuningApp(
        FineTuningAppProps(
            state_location=get_state_location(),
            datasets_manager=DatasetsManagerSimple(),
            models_manager=ModelsManagerSimple(),
            jobs_manager=FineTuningJobsManagerSimple(),
            mlflow_manager=MLflowEvaluationJobsManagerSimple()
        )
    )

# Module for setting up navigation


def setup_navigation():
    pg = st.navigation([
        st.Page("pgs/home.py", title="Home"),
        st.Page("pgs/datasets.py", title="Import Datasets"),
        st.Page("pgs/view_datasets.py", title="View Datasets"),
        st.Page("pgs/models.py", title="Import Base Models"),
        st.Page("pgs/view_models.py", title="View Base Models"),
        st.Page("pgs/prompts.py", title="Training Prompts"),
        st.Page("pgs/train_adapter.py", title="Train a New Adapter"),
        st.Page("pgs/jobs.py", title="Training Job Tracking"),
        st.Page("pgs/evaluate.py", title="Local Adapter Comparison"),
        st.Page("pgs/mlflow.py", title="Run MLFlow Evaluation"),
        st.Page("pgs/mlflow_jobs.py", title="View MLflow Runs"),
        st.Page("pgs/export.py", title="Export to CML Model Registry"),
        st.Page("pgs/deploy.py", title="Deploy to Cloudera AI Inference"),
    ], position="hidden")
    pg.run()

# Module for sidebar content


def setup_sidebar():
    with st.sidebar:
        st.image("./resources/images/ft-logo.png")
        st.subheader("")
        st.markdown("Navigation")
        st.page_link("pgs/home.py", label="Home", icon=":material/home:")
        st.write("\n")

        st.markdown("AI Toolkit")
        st.page_link("pgs/datasets.py", label="Import Datasets", icon=":material/publish:")
        st.page_link("pgs/view_datasets.py", label="View Datasets", icon=":material/data_object:")
        st.page_link("pgs/models.py", label="Import Base Models", icon=":material/neurology:")
        st.page_link("pgs/view_models.py", label="View Base Models", icon=":material/view_day:")
        st.page_link("pgs/prompts.py", label="Training Prompts", icon=":material/chat:")
        st.write("\n")

        st.markdown("Experiments")
        st.page_link("pgs/train_adapter.py", label="Train a New Adapter", icon=":material/forward:")
        st.page_link("pgs/jobs.py", label="Monitor Training Jobs", icon=":material/subscriptions:")
        st.page_link("pgs/evaluate.py", label="Local Adapter Comparison", icon=":material/difference:")
        st.page_link("pgs/mlflow.py", label="Run MLFlow Evaluation", icon=":material/model_training:")
        st.page_link("pgs/mlflow_jobs.py", label="View MLflow Runs", icon=":material/monitoring:")
        st.write("\n")

        st.markdown("CML")
        st.page_link("pgs/export.py", label="Export to CML Model Registry", icon=":material/move_group:")
        st.page_link("pgs/deploy.py", label="Deploy to Cloudera AI Inference", icon=":material/deployed_code:")
        st.subheader("", divider="green")

        project_owner = get_env_variable('PROJECT_OWNER', 'User')
        cdsw_url = get_env_variable('CDSW_DOMAIN', 'CDSW Url')
        st.page_link(f"https://{cdsw_url}", label=f"{project_owner}", icon=":material/account_circle:")


# Main function to orchestrate the setup
st.set_page_config(
    page_title="LLM Finetuning Studio",
    page_icon="./resources/images/architecture_24dp_EA3323_FILL0_wght400_GRAD0_opsz48.png",
    layout="wide"
)
apply_custom_css()
initialize_ft_app()
setup_navigation()
setup_sidebar()
