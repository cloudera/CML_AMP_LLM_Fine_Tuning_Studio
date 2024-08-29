import streamlit as st
from pgs.streamlit_utils import get_fine_tuning_studio_client
from ft.consts import IconPaths, DIVIDER_COLOR

# Instantiate the client to the FTS gRPC app server.
fts = get_fine_tuning_studio_client()


def display_page_header():
    with st.container(border=True):
        col1, col2 = st.columns([1, 17])
        with col1:
            col1.image(IconPaths.CML.EXPORT_TO_CML_MODEL_REGISTRY)
        with col2:
            col2.subheader('Deploy to Cloudera AI Inference', divider=DIVIDER_COLOR)
            st.caption("Deploy your models to Cloudera AI Inference for production use and real-world application deployment.")


def display_page():
    st.info("Coming soon !", icon=":material/info:")


display_page_header()
display_page()
