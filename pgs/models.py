import streamlit as st
from ft.state import get_state
from ft.app import get_app
from ft.model import ModelMetadata, ModelType, ImportModelRequest
from ft.adapter import AdapterMetadata, AdapterType
from typing import List


def display_header():
    with st.container(border=True):
        col1, col2 = st.columns([1, 13])
        with col1:
            col1.image("./resources/images/neurology_24dp_EA3323_FILL0_wght400_GRAD0_opsz40.png")
        with col2:
            col2.subheader('Base Models & Adapters', divider="orange")
            st.write("Import foundational LLM models from Hugging Face or local sources to align with your fine-tuning job specific requirements.")

def display_import_section():
    st.write("\n")
    st.markdown('''
    <style>
    .katex-html {
        text-align: left;
    }
    </style>''', unsafe_allow_html=True)
    st.latex(r'\textsf{\large Import Models}')

    with st.container(border=True):
        upload_ct = st.container()
        import_hf_tab, upload_tab = upload_ct.tabs(["**Huggingface**", "**Local**"])

        with import_hf_tab:
            display_huggingface_import()
        with upload_tab:
            st.info("Feature coming soon")

def display_huggingface_import():
    col1, col2 = st.columns([4, 1])
    import_hf_dataset_name = col1.text_input('Model Name', placeholder="huggingface/model", label_visibility='collapsed')
    import_hf_dataset = col2.button("Import", type="primary", use_container_width=True)

    if import_hf_dataset:
        if import_hf_dataset_name:
            with st.spinner("Loading Model..."):
                get_app().import_model(ImportModelRequest(
                    type=ModelType.HUGGINGFACE,
                    huggingface_name=import_hf_dataset_name
                ))
        else:
            st.error("Please enter a model name.")

def display_models_section():
    st.latex(r'\textsf{\large Available Models}')
    models: List[ModelMetadata] = get_state().models
    adapters: List[AdapterMetadata] = get_state().adapters

    with st.container(border=True):
        tab1, tab2 = st.tabs(["**Huggingface Models**", "**Local Models**"])
        with tab1:
            display_models([model for model in models if model.type == ModelType.HUGGINGFACE], adapters)
        with tab2:
            display_models([model for model in models if model.type != ModelType.HUGGINGFACE], adapters)

def display_models(models: List[ModelMetadata], adapters: List[AdapterMetadata]):
    if not models:
        st.info("No models available.")
        return

    cont = st.container()
    for i, model in enumerate(models):
        if i % 2 == 0:
            col1, col2 = cont.columns(2)
        st.write("\n")

        ds_cont = col1 if i % 2 == 0 else col2
        with ds_cont.container(border=True):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**{model.name}**")
            c1.caption(model.id)

            remove = c2.button("Remove", type="primary", key=f"{model.id}_remove", use_container_width=True)

            if remove:
                get_app().remove_model(model.id)
                st.rerun()

            model_adapters = [adapter for adapter in adapters if adapter.model_id == model.id]
            expander = ds_cont.expander("Adapters")
            for adapter in model_adapters:
                display_adapter(adapter, expander)

            add_adapter_button = expander.button("Add Adapter", type="primary", key=f"{model.id}_add_adapter", use_container_width=True)
            if add_adapter_button:
                st.toast("You can't do that yet.", icon=":material/info:")

def display_adapter(adapter: AdapterMetadata, container):
    with container.container(border=True):
        c1, c2 = container.columns([4, 1])
        c1.text(adapter.name)
        if adapter.type == AdapterType.LOCAL:
            c1.caption(adapter.location)

        remove = c2.button("Remove", type="secondary", key=f"{adapter.id}_remove", use_container_width=True)

        if remove:
            st.toast("You can't do that yet.", icon=":material/info:")

display_header()
display_import_section()
st.markdown("---")
display_models_section()
