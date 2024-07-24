import streamlit as st
from ft.state import get_state
from ft.app import get_app
from ft.model import *
from ft.adapter import *
from typing import List, Dict, Optional

st.subheader("Import Models")


with st.container():
    upload_ct = st.container(border=True)
    import_hf_tab, upload_tab = upload_ct.tabs(["Huggingface", "Local"])

    c1, c2 = import_hf_tab.columns([4, 1])
    import_hf_dataset_name = c1.text_input('Model Name', placeholder="huggingface/model", label_visibility='collapsed')
    import_hf_dataset = c2.button("Import", type="primary", use_container_width=True)

    if import_hf_dataset:
        with st.spinner("Loading Model..."):
            get_app().import_model(ImportModelRequest(
                type=ModelType.HUGGINGFACE,
                huggingface_name=import_hf_dataset_name
            ))

    upload_tab.info("Feature coming soon")

st.markdown("---")

st.subheader("Available Models")

models: List[ModelMetadata] = get_state().models
adapters: List[AdapterMetadata] = get_state().adapters

cont = st.container(border=False)

tab1, tab2 = st.tabs(["Huggingface Models", "Local Models"])

def display_models(models):
    cont = st.container(border=False)
    for i, model in enumerate(models):
        if i % 2 == 0:
            col1, col2 = cont.columns(2)
        st.write("\n")
    
        ds_cont = col1 if i % 2 == 0 else col2
        
        with ds_cont.container(border=True):
            c1, c2 = st.columns([5, 1])
            c1.subheader(model.name)
            c1.caption(model.id)

            remove = c2.button("Remove", type="primary", key=f"{model.id}_remove", use_container_width=True)

            if remove:
                get_app().remove_model(model.id)
                st.rerun()

            model_adapters = list(filter(lambda x: x.model_id == model.id, adapters))
            expander = ds_cont.expander("Adapters")
            
            for model_adapter in model_adapters:
                cc = expander.container(border=True)

                cc1, cc2 = expander.columns([5, 1])
                cc1.text(model_adapter.name)
                if model_adapter.type == AdapterType.LOCAL:
                    cc1.caption(model_adapter.location)

                remove = cc2.button("Remove", type="primary", key=f"{model_adapter.id}_remove", use_container_width=True)

                if remove: 
                    st.toast("You can't do that yet.")

            add_adapter_button = expander.button("Add Adapter", type="primary", key=f"{model.id}_add_adapter", use_container_width=True)
            if add_adapter_button:
                st.toast("You can't do that yet.")

with tab1:
    huggingface_models = [model for model in models if model.type == ModelType.HUGGINGFACE]
    if not huggingface_models:
        st.info("No Huggingface models available.")
    else:
        display_models(huggingface_models)

with tab2:
    custom_models = [model for model in models if model.type != ModelType.HUGGINGFACE]
    if not custom_models:
        st.info("No Local models available.")
    else:
        display_models(custom_models)
