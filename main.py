import streamlit as st
import ft.app
from ft.app import create_app, FineTuningAppProps, FineTuningApp
from ft.state import get_state_location
from ft.managers import (
    DatasetsManagerSimple,
    ModelsManagerSimple,
    FineTuningJobsManagerSimple
)

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
    st.Page("pgs/home.py", title="Home", icon=":material/home:"),
    st.Page("pgs/datasets.py", title="Datasets", icon=":material/database:"),
    st.Page("pgs/models.py", title="Models and Adapters", icon=":material/neurology:"),
    st.Page("pgs/prompts.py", title="Prompts", icon=":material/chat:"),
    st.Page("pgs/train_adapter.py", title="Train a New Adapter", icon=":material/forward:"),
    st.Page("pgs/jobs.py", title="Training Jobs", icon=":material/subscriptions:"),
    st.Page("pgs/evaluate.py", title="Local Evaluation", icon=":material/difference:")
])
pg.run()

