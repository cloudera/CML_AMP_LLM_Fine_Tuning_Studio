import streamlit as st
import streamlit as st
from ft.state import get_state
from ft.pipeline import fetch_pipeline
from ft.api import *
import mlflow
from transformers import GenerationConfig
from typing import List


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
            current_models = get_state().models
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
            model_adapters = get_state().adapters
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
            # TODO: move model export logic out of Streamlit UI logic
            with st.spinner("Generating an MLFlow model pipeline..."):
                model: ModelMetadata = current_models[model_idx]
                adapter: AdapterMetadata = model_adapters[adapter_idx]

                # For now, let's assume HF model is available. If not, we should be ideally
                # raising an error or handling custom models differently.
                adapter_location_or_name = adapter.location if adapter.type == AdapterType.ADAPTER_TYPE_PROJECT else adapter.huggingface_name
                pipeline = fetch_pipeline(
                    model_name=model.huggingface_model_name,
                    adapter_name=adapter_location_or_name)

            signature = mlflow.models.infer_signature(
                model_input="What are the three primary colors?",
                model_output="The three primary colors are red, yellow, and blue.",
            )

            # TODO: pull out generation config to arguments
            config = GenerationConfig(
                do_sample=True,
                temperature=0.8,
                max_new_tokens=60,
                top_p=1
            )

            with st.spinner("Logging model to MLFlow..."):
                with mlflow.start_run():
                    model_info = mlflow.transformers.log_model(
                        transformers_model=pipeline,
                        torch_dtype='float16',
                        artifact_path="custom-pipe",        # artifact_path can be dynamic
                        signature=signature,
                        registered_model_name=registered_model_name,  # model_name can be dynamic
                        model_config=config.to_dict()
                    )

            st.success(
                "Exported Model. Please go to **CML Model Registry** to view your model!",
                icon=":material/check:")
            st.toast("Exported Model. Please go to **CML Model Registry** to view your model!", icon=":material/check:")
        except Exception as e:
            st.error(f"Failed to export model: **{str(e)}**", icon=":material/error:")
            st.toast(f"Failed to export model: **{str(e)}**", icon=":material/error:")
