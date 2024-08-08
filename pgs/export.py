import streamlit as st
import streamlit as st
from ft.api import *
from typing import List
from pgs.streamlit_utils import get_fine_tuning_studio_client

# Instantiate the client to the FTS gRPC app server.
fts = get_fine_tuning_studio_client()

with st.container(border=True):
    col1, col2 = st.columns([1, 17])
    with col1:
        col1.image("./resources/images/move_group_24dp_EA3323_FILL0_wght400_GRAD0_opsz40.png")
    with col2:
        col2.subheader('Export to CML Model Registry', divider='red')
        st.caption("Export your fine-tuned models and adapters to the Cloudera Model Registry for easy access and deployment.")

# Container for model and adapter selection
with st.container(border=True):
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            registered_model_name = st.text_input(
                "Registered Model Name",
                placeholder="human-friendly name in model registry",
                key="registered_model_name")

        with col2:
            current_models = fts.get_models()
            model_idx = st.selectbox(
                "Base Model",
                range(
                    len(current_models)),
                format_func=lambda x: current_models[x].name,
                index=None)

    # Container for dataset and prompt selection
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            model_description = st.text_input(
                "Model Description",
                placeholder="description of the model",
                key="model_description")

        with col2:
            model_adapters = fts.get_adapters()
            if model_idx is not None:
                current_model: ModelMetadata = current_models[model_idx]
                model_adapters: List[AdapterMetadata] = list(
                    filter(lambda x: x.model_id == current_model.id, model_adapters))
            else:
                model_adapters = []

            adapter_idx = st.selectbox("Choose an Adapter", range(len(model_adapters)),
                                       format_func=lambda x: model_adapters[x].name, index=None)

    # Start job button
    button_enabled = model_idx is not None and adapter_idx is not None and registered_model_name != ""
    start_job_button = st.button(
        "Register Model to CML Model Registry",
        type="primary",
        use_container_width=True,
        disabled=not button_enabled)

    if start_job_button:
        try:
            with st.spinner("Exporting model to CML Model Registry..."):
                res: ExportModelRequest = fts.ExportModel(ExportModelRequest(
                    type=ModelType.MODEL_TYPE_MODEL_REGISTRY,
                    model_id=current_models[model_idx].id,
                    adapter_id=model_adapters[adapter_idx].id,
                    model_name=registered_model_name,
                ))
            st.success(
                "Exported Model. Please go to **CML Model Registry** to view your model!",
                icon=":material/check:")
            st.toast("Exported Model. Please go to **CML Model Registry** to view your model!", icon=":material/check:")
        except Exception as e:
            st.error(f"Failed to export model: **{str(e)}**", icon=":material/error:")
            st.toast(f"Failed to export model: **{str(e)}**", icon=":material/error:")
