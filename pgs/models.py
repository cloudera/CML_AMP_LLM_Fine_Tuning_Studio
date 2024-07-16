import streamlit as st
from ft.state import get_state
from ft.app import get_app
from ft.model import *
from ft.adapter import *
from typing import List, Dict, Optional

st.title("Models")


with st.container(border=True):
    st.header("Import Model")

    upload_ct = st.container(border=True)
    import_hf_tab, upload_tab = upload_ct.tabs(["Huggingface", "Local"])

    import_hf_tab.subheader("Import Huggingface Model")
    c1, c2 = import_hf_tab.columns([4, 1])
    import_hf_dataset_name = c1.text_input('Model Name', placeholder="huggingface/model", label_visibility='collapsed')
    import_hf_dataset = c2.button("Import", use_container_width=True)

    if import_hf_dataset:
        with st.spinner("Loading Model..."):
            get_app().import_model(ImportModelRequest(
                type=ModelType.HUGGINGFACE,
                huggingface_name=import_hf_dataset_name
            ))

    upload_tab.header("Upload Model")
    upload_tab.caption("Feature coming soon")





st.header("Available Models")

models: List[ModelMetadata] = get_state().models
adapters: List[AdapterMetadata] = get_state().adapters

cont = st.container(border=False)

for model in models:
    ds_cont = cont.container(border=True)
    
    if model.type == ModelType.HUGGINGFACE:
        c1, c2 = ds_cont.columns([6, 1])
        c1.subheader(model.name)
        c1.caption(model.id)

        remove = c2.button("Remove", key=f"{model.id}_remove", use_container_width=True)

        if remove:
            get_app().remove_model(model.id)
            st.rerun()

    model_adapters: List[AdapterMetadata] = filter(lambda x: x.model_id == model.id, adapters)
    expander = ds_cont.expander("Adapters")
    
    # expander.subheader("Adapters:")
    for model_adapter in model_adapters:
        cc = expander.container(border=True)

        cc1, cc2 = cc.columns([5, 1])
        cc1.text(model_adapter.name)
        if model_adapter.type == AdapterType.LOCAL:
            cc1.caption(model_adapter.location)

        remove = cc2.button("Remove", key=f"{model_adapter.id}_remove", use_container_width=True)

        if remove: 
            st.toast("You can't do that yet.")

    add_adapter_button = expander.button("Add Adapter", key=f"{model.id}_add_adapter", use_container_width=True)
    if add_adapter_button:
        # TODO: pull in local adapters
        st.toast("You can't do that yet.")
