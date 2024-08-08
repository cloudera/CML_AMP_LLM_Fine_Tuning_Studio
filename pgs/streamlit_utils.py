import streamlit as st

from ft.client import FineTuningStudioClient
from cmlapi import CMLServiceApi, default_client


@st.cache_resource
def get_fine_tuning_studio_client() -> FineTuningStudioClient:
    """
    Returns the FTS client, which is a wrapper to the
    gRPC stub to connect to the gRPC server. Because we only
    need one client connection to the gRPC server, we can share
    this client on the streamlit server.
    """

    client = FineTuningStudioClient()
    return client


@st.cache_resource
def get_cml_client() -> CMLServiceApi:
    """
    Return a CML fts.
    """

    client = default_client()
    return client
