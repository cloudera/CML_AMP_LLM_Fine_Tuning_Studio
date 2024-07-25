import streamlit as st
import ft.app
from ft.app import create_app, FineTuningAppProps, FineTuningApp
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
    layout="wide"  # Keep layout as wide
)

def apply_custom_css_for_tab():
    css = '''
    <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1rem;
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

# Set the instance for the app. NOTE: due to streamlit application
# ecosystem, this is technically recreated upon every interaction
# with the streamlit app. This might not be a good design pattern
# if the app needs to hold on to any persistent connections. But for now,
# the app is just a collection of utility classes that perform some
# form of work on the application's state (project-level session data).

ft.app.INSTANCE = FineTuningApp(
    FineTuningAppProps(
            state_location=get_state_location(),
            datasets_manager=DatasetsManagerSimple(),
            models_manager=ModelsManagerSimple(),
            jobs_manager=FineTuningJobsManagerSimple()
        )
)

pg = st.navigation([
    st.Page("pgs/home.py", title="Home"),
    st.Page("pgs/datasets.py", title="Import Datasets"),
    st.Page("pgs/view_datasets.py", title="View Datasets"),
    st.Page("pgs/models.py", title="Base Models and Adapters"),
    st.Page("pgs/prompts.py", title="Training Prompts"),
    st.Page("pgs/train_adapter.py", title="Train a New Adapter"),
    st.Page("pgs/jobs.py", title="Training Job Tracking"),
    st.Page("pgs/evaluate.py", title="Local Adapater Comparison"),
    st.Page("pgs/export.py", title="Export to CML Model Registry"),
    st.Page("pgs/deploy.py", title="Deploy to Cloudera AI Inference"),    
], position="hidden")

pg.run()

with st.sidebar:
    
    st.markdown("**Navigation**")
    st.page_link("pgs/home.py", label="Home", icon=":material/home:")
    st.write("\n")

    st.markdown("**Datasets**")
    st.page_link("pgs/datasets.py", label="Import Datasets", icon=":material/publish:")
    st.page_link("pgs/view_datasets.py", label="View Datasets", icon=":material/data_object:")
    st.write("\n")

    st.markdown("**Models & Adapters**")
    st.page_link("pgs/models.py", label="Base Models and Adapters", icon=":material/neurology:")
    st.page_link("pgs/prompts.py", label="Training Prompts", icon=":material/chat:")
    st.write("\n")

    st.markdown("**Experiments**")
    st.page_link("pgs/train_adapter.py", label="Train a New Adapter", icon=":material/forward:")
    st.page_link("pgs/jobs.py", label="Monitor Training Jobs", icon=":material/subscriptions:")
    st.page_link("pgs/evaluate.py", label="Local Adapter Comparison", icon=":material/difference:")
    st.write("\n")

    st.markdown("**Model Management**")
    st.page_link("pgs/export.py", label="Export to CML Model Registry", icon=":material/move_group:")
    st.page_link("pgs/deploy.py", label="Deploy to Cloudera AI Inference", icon=":material/deployed_code:")
    st.write("\n")

apply_custom_css_for_tab()