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
    page_icon="./resources/images/diapason-tuner-svgrepo-com.svg",
    layout="wide"  # Keep layout as wide
)

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

pg = st.navigation({
    "Navigation": [
        st.Page("pgs/home.py", title="Home", icon=":material/home:"),
    ],
    "Datasets": [
        st.Page("pgs/datasets.py", title="Import Datasets", icon=":material/database:"),
        st.Page("pgs/view_datasets.py", title="View Datasets", icon=":material/database:"),
    ],
    "Models & Adapaters": [
        st.Page("pgs/models.py", title="Models and Adapters", icon=":material/neurology:"),
        st.Page("pgs/prompts.py", title="Training Prompts", icon=":material/chat:"),
    ],
    "Experiments": [
        st.Page("pgs/train_adapter.py", title="Train a New Adapter", icon=":material/forward:"),
        st.Page("pgs/jobs.py", title="Training Job Tracking", icon=":material/subscriptions:"),
        st.Page("pgs/evaluate.py", title="Local Adapater Comparison", icon=":material/difference:")
    ],
    "Model Management": [
        st.Page("pgs/export.py", title="Export to CML Model Registry", icon=":material/upgrade:"),
        st.Page("pgs/deploy.py", title="Deploy to Cloudera AI Inference", icon=":material/deployed_code:"),
    ]
})

pg.run()

