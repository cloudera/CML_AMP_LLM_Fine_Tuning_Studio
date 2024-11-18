import streamlit as st
import streamlit as st
from ft.api import *
from typing import List
from pgs.streamlit_utils import get_fine_tuning_studio_client
from ft.consts import IconPaths, DIVIDER_COLOR, DEFAULT_GENERATIONAL_CONFIG
import json

MODEL_EXPORT_TYPE_MAP = {
    "Model Registry": ModelExportType.MODEL_REGISTRY,
    "CML Model": ModelExportType.CML_MODEL
}

# Instantiate the client to the FTS gRPC app server.
fts = get_fine_tuning_studio_client()


def create_header():
    with st.container(border=True):
        col1, col2 = st.columns([1, 17])
        with col1:
            col1.image(IconPaths.CML.EXPORT_TO_CML_MODEL_REGISTRY)
        with col2:
            col2.subheader('Export/Deploy Model', divider=DIVIDER_COLOR)
            st.caption(
                "Export your fine-tuned models and adapters to Cloudera Model Registry or deploy to CML Models for use in production.")


def handle_export_or_deploy():
    with st.container():
        deploy_cml_models_tab, export_mr_tab = st.tabs(["**CML Model Deployment**", "**CML Model Registry**"])

        with deploy_cml_models_tab:
            display_model_deployment_tab()
        with export_mr_tab:
            display_mr_tab()


# Container for model and adapter selection
def display_mr_tab():
    with st.container(border=True):
        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                registered_model_name = st.text_input(
                    "Model Name",
                    placeholder="human-friendly name in model registry",
                    key="registered_model_name")

            with col2:
                current_models = fts.get_models()
                model_idx = st.selectbox(
                    "Base Model",
                    range(
                        len(current_models)),
                    format_func=lambda x: current_models[x].name,
                    index=None,
                    key="base_model_idx_mr")

        # Container for dataset and prompt selection
        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                model_description = st.text_input(
                    "Model Description",
                    placeholder="description of the model",
                    key="model_description_mr")

            with col2:
                model_adapters = fts.get_adapters()
                if model_idx is not None:
                    current_model: ModelMetadata = current_models[model_idx]
                    model_adapters: List[AdapterMetadata] = list(
                        filter(lambda x: x.model_id == current_model.id, model_adapters))
                else:
                    model_adapters = []

                adapter_idx = st.selectbox(
                    "Choose an Adapter",
                    range(
                        len(model_adapters)),
                    format_func=lambda x: model_adapters[x].name,
                    index=None,
                    key="adapter_idx_mr")

        # Start job button
        button_enabled = model_idx is not None and adapter_idx is not None and registered_model_name != ""
        if button_enabled:
            gen_config_text = st.text_area(
                "Generational Config",
                json.dumps(
                    DEFAULT_GENERATIONAL_CONFIG,
                    indent=2),
                height=200,
                key="gen_config_text_mr")
        start_job_button = st.button(
            "Export Model",
            type="primary",
            use_container_width=True,
            disabled=not button_enabled,
            key="start_job_button_mr")

        if start_job_button:
            try:
                with st.spinner("Exporting model to CML Model Registry..."):
                    res: ExportModelRequest = fts.ExportModel(ExportModelRequest(
                        type=ModelExportType.MODEL_REGISTRY,
                        base_model_id=current_models[model_idx].id,
                        adapter_id=model_adapters[adapter_idx].id,
                        model_name=registered_model_name,
                        model_description=model_description,
                        generation_config=gen_config_text,
                    ))
                st.success(
                    "Model export is started. Please go to **CML Model Registry** to view progress.",
                    icon=":material/check:")
                st.toast(
                    "Model export is started. Please go to **CML Model Registry** to view progress.",
                    icon=":material/check:")
            except Exception as e:
                st.error(f"Failed to export model: **{str(e)}**", icon=":material/error:")
                st.toast(f"Failed to export model: **{str(e)}**", icon=":material/error:")


def display_model_deployment_tab():
    # Container for model and adapter selection
    with st.container(border=True):
        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                deployed_model_name = st.text_input(
                    "Model Name",
                    placeholder="human-friendly name for deployment",
                    key="deployed_model_name")

            with col2:
                current_models = fts.get_models()
                model_idx = st.selectbox(
                    "Base Model",
                    range(
                        len(current_models)),
                    format_func=lambda x: current_models[x].name,
                    index=None,
                    key="base_model_idx_cml_deploy")

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

                adapter_idx = st.selectbox(
                    "Choose an Adapter",
                    range(
                        len(model_adapters)),
                    format_func=lambda x: model_adapters[x].name,
                    index=None,
                    key="adapter_idx_cml_deploy")

        # Start job button
        button_enabled = model_idx is not None and adapter_idx is not None and deployed_model_name != ""
        if button_enabled:
            gen_config_text = st.text_area(
                "Generational Config",
                json.dumps(
                    DEFAULT_GENERATIONAL_CONFIG,
                    indent=2),
                height=200,
                key="gen_config_text_cml_deploy")
        start_job_button = st.button(
            "Deploy Model",
            type="primary",
            use_container_width=True,
            disabled=not button_enabled,
            key="start_job_button_cml_deploy")

        if start_job_button:
            try:
                with st.spinner("Deploying model to CML Model Deployments..."):
                    res: ExportModelRequest = fts.ExportModel(ExportModelRequest(
                        type=ModelExportType.CML_MODEL,
                        base_model_id=current_models[model_idx].id,
                        adapter_id=model_adapters[adapter_idx].id,
                        model_name=deployed_model_name,
                        model_description=model_description,
                        generation_config=gen_config_text,
                    ))
                st.success(
                    "Model deployment is started. Please go to **Model Deployments** to view progress.",
                    icon=":material/check:")
                st.toast(
                    "Model deployment is started. Please go to **Model Deployments** to view progress.",
                    icon=":material/check:")
            except Exception as e:
                st.error(f"Failed to deploy model: **{str(e)}**", icon=":material/error:")
                st.toast(f"Failed to deploy model: **{str(e)}**", icon=":material/error:")


create_header()
handle_export_or_deploy()
