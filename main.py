import streamlit as st
import ft.app
from ft.app import FineTuningAppProps, FineTuningApp
from ft.state import get_state_location
from ft.managers import (
    DatasetsManagerSimple,
    ModelsManagerSimple,
    FineTuningJobsManagerSimple
)
import streamlit.components.v1 as components

# Set the page configuration
st.set_page_config(
    page_title="LLM Finetuning Studio",
    page_icon="./resources/images/architecture_24dp_EA3323_FILL0_wght400_GRAD0_opsz48.png",
    layout="wide"  # Set layout to wide
)


def apply_custom_css_for_tab():
    """
    Apply custom CSS for the tab styling to adjust font sizes.
    """
    css = '''
    <style>
        h3 {
            font-size: 1.1rem;  /* Adjust the font size as needed */
        }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 0.9rem;
        }
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)


st.markdown("""
    <style>
    [data-testid="stHeader"] {
        color: #4CAF50; /* Change text color */
        background-color: #f0f0f0; /* Change background color */
    }
    /* Target the sidebar content using data-testid attribute */
    [data-testid="stSidebarContent"] {
        background: #132329;  /* Shiny black gradient */
        color: white;         /* Change text color to white for better contrast */
    }
    /* Ensure all text elements inside the sidebar are white */
    [data-testid="stSidebarContent"] * {
        color: white;         /* Change text color to white */
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize the FineTuningApp instance
ft.app.INSTANCE = FineTuningApp(
    FineTuningAppProps(
        state_location=get_state_location(),
        datasets_manager=DatasetsManagerSimple(),
        models_manager=ModelsManagerSimple(),
        jobs_manager=FineTuningJobsManagerSimple()
    )
)

# Navigation setup for the application
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
    st.Page("pgs/export.py", title="Export to CML Model Registry"),
    st.Page("pgs/deploy.py", title="Deploy to Cloudera AI Inference"),
], position="hidden")

pg.run()

with st.sidebar:
    st.subheader("")
    st.markdown("Navigation")
    st.page_link("pgs/home.py", label="Home", icon=":material/home:")
    st.subheader("")

    st.markdown("Studio")
    st.page_link("pgs/datasets.py", label="Import Datasets", icon=":material/publish:")
    st.page_link("pgs/view_datasets.py", label="View Datasets", icon=":material/data_object:")
    st.page_link("pgs/models.py", label="Import Base Models", icon=":material/neurology:")
    st.page_link("pgs/view_models.py", label="View Base Models", icon=":material/view_day:")
    st.page_link("pgs/prompts.py", label="Training Prompts", icon=":material/chat:")
    st.page_link("pgs/train_adapter.py", label="Train a New Adapter", icon=":material/forward:")
    st.page_link("pgs/jobs.py", label="Monitor Training Jobs", icon=":material/subscriptions:")
    st.page_link("pgs/evaluate.py", label="Local Adapter Comparison", icon=":material/difference:")
    st.page_link("pgs/mlflow.py", label="Run MLFlow Evaluation", icon=":material/model_training:")
    st.page_link("pgs/export.py", label="Export to CML Model Registry", icon=":material/move_group:")
    st.page_link("pgs/deploy.py", label="Deploy to Cloudera AI Inference", icon=":material/deployed_code:")

apply_custom_css_for_tab()
