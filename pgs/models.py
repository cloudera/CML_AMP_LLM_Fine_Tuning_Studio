import streamlit as st
from ft.api import *
from typing import List
from cmlapi import RegisteredModelDetails
from pgs.streamlit_utils import get_fine_tuning_studio_client, get_cml_client
from ft.consts import IconPaths, DIVIDER_COLOR

# Instantiate the client to the FTS gRPC app server.
fts = get_fine_tuning_studio_client()
cml = get_cml_client()


def display_header():
    with st.container(border=True):
        col1, col2 = st.columns([1, 17])
        with col1:
            col1.image(IconPaths.AIToolkit.IMPORT_BASE_MODELS)
        with col2:
            col2.subheader('Import Base Models', divider=DIVIDER_COLOR)
            st.caption(
                "Import foundational LLM models from Hugging Face or local sources to align with your fine-tuning job specific requirements.")


def display_import_section():
    with st.container():
        upload_ct = st.container()
        import_hf_tab, registry_tab, upload_tab = upload_ct.tabs(
            ["**Import Huggingface Models**", "**Import from Model Registry**", "**Upload from Project Files**"])

        with import_hf_tab:
            display_huggingface_import()
        with registry_tab:
            st.info("Coming soon !", icon=":material/info:")
        with upload_tab:
            st.info("Coming soon !", icon=":material/info:")


def display_model_registry_import():

    # List the available models in the model registry.
    model_registry_models: List[RegisteredModelDetails] = cml.list_registered_models().models

    if not model_registry_models:
        st.info("There are no registered models in this workspace.")
    else:

        col1, col2 = st.columns([4, 1])
        model_idx = col1.selectbox(
            "Registered models",
            range(len(model_registry_models)),
            format_func=lambda x: model_registry_models[x].name,
            index=None,
            label_visibility="collapsed"
        )
        import_registered_model = col2.button("Import", type="primary", use_container_width=True, key="registry_import")

        if import_registered_model:
            if model_idx is not None:
                with st.spinner("Loading Model..."):
                    try:
                        fts.AddModel(
                            AddModelRequest(
                                type=ModelType.MODEL_REGISTRY,
                                model_registry_id=model_registry_models[model_idx].model_id
                            )
                        )
                        st.success(
                            "Model imported successfully. Please check **View Models** page!",
                            icon=":material/check:")
                    except Exception as e:
                        st.error(f"Error importing model: {str(e)}", icon=":material/error:")
            else:
                st.error("Please enter a model name.", icon=":material/info:")

    st.write("\n")

    st.info("""
        **Model Registry Models:**

        Users can also import models that exist within a CML Model Registry tied to this workspace.

    """, icon=":material/info:")


def display_huggingface_import():
    col1, col2 = st.columns([4, 1])
    import_hf_model_name = col1.text_input(
        'Model Name',
        placeholder="organization/model",
        label_visibility='collapsed')
    import_hf_model = col2.button("Import", type="primary", use_container_width=True)

    if import_hf_model:
        if import_hf_model_name:
            with st.spinner("Loading Model..."):
                try:
                    fts.AddModel(
                        AddModelRequest(
                            type=ModelType.HUGGINGFACE,
                            huggingface_name=import_hf_model_name
                        )
                    )
                    st.success(
                        "Model imported successfully. Please check **View Models** page!",
                        icon=":material/check:")
                except Exception as e:
                    st.error(f"Error importing model: {str(e)}", icon=":material/error:")
        else:
            st.error("Please enter a model name.", icon=":material/info:")

    st.write("\n")

    st.info("""
        **Hugging Face Models:**

        Hugging Face is a platform that hosts a wide range of machine learning models, including transformers for natural language processing tasks.

        **How to Import Foundational Models from Hugging Face:**
        - Enter the model name in the format `organization/model` in the input field provided.
        - Click the "Import" button to load the model.
        - Once imported, the model will be available for use in your application.

        **Finding the Organization and Model Name:**
        - Visit the Hugging Face Models [website](https://huggingface.co/models).
        - Browse or search for the model you need.
        - The model name will be in the format `organization/model` on the model's page.

        **Usage in Training Adapters:**
        - Imported foundational models can be fine-tuned using adapters.
        - Adapters allow you to customize the model's performance for specific tasks without modifying the original model weights.
        - This approach is efficient and enables rapid deployment of specialized models.


    """, icon=":material/info:")


display_header()
display_import_section()
