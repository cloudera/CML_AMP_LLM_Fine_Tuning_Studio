import streamlit as st
from ft.api import *
from typing import List
from pgs.streamlit_utils import get_fine_tuning_studio_client

# Instantiate the client to the FTS gRPC app server.
fts = get_fine_tuning_studio_client()


def display_header():
    with st.container(border=True):
        col1, col2 = st.columns([1, 17])
        with col1:
            col1.image("./resources/images/view_day_24dp_EA3323_FILL0_wght400_GRAD0_opsz48.png")
        with col2:
            col2.subheader('Base Models & Adapters', divider='red')
            st.caption("Review the imported base models and the adapters generated during the fine-tuning process. Note that a single model can contain multiple adapters.")


def display_models_section():
    models: List[ModelMetadata] = fts.get_models()
    adapters: List[AdapterMetadata] = fts.get_adapters()

    with st.container():
        tab1, tab2, tab3 = st.tabs(["**Huggingface**", "**Model Registry**", "**Project**"])
        with tab1:
            display_models([model for model in models if model.type == ModelType.MODEL_TYPE_HUGGINGFACE], adapters)
        with tab2:
            display_models([model for model in models if model.type == ModelType.MODEL_TYPE_MODEL_REGISTRY], adapters)
        with tab3:
            display_models([model for model in models if model.type == ModelType.MODEL_TYPE_PROJECT], adapters)


def display_models(models: List[ModelMetadata], adapters: List[AdapterMetadata]):
    if not models:
        st.info("No models available.", icon=":material/info:")
        return

    cont = st.container()
    for i, model in enumerate(models):
        if i % 2 == 0:
            col1, col2 = cont.columns(2)
        st.write("\n")

        s_cont = col1 if i % 2 == 0 else col2
        ds_cont = s_cont.container(border=True)
        with ds_cont:
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**{model.name}**")
            c1.caption(f"**{model.id}**")

            remove = c2.button("Remove", type="primary", key=f"{model.id}_remove", use_container_width=True)

            if remove:
                fts.RemoveModel(
                    RemoveModelRequest(
                        id=model.id
                    )
                )
                st.rerun()

            model_adapters = [adapter for adapter in adapters if adapter.model_id == model.id]
            expander = ds_cont.expander("Adapters")

            if not model_adapters:
                expander.info(
                    "No available adpater to display. Please go to **Train a new Adapater** page.",
                    icon=":material/info:")
            for adapter in model_adapters:
                display_adapter(adapter, expander)


def display_adapter(adapter: AdapterMetadata, container):
    with container:
        c1, c2 = container.columns([4, 1])
        c1.text(adapter.name)
        if adapter.type == AdapterType.ADAPTER_TYPE_PROJECT:
            c1.caption(adapter.location)
        elif adapter.type == AdapterType.ADAPTER_TYPE_HUGGINGFACE:
            c1.caption(adapter.huggingface_name)

        remove = c2.button("Remove", type="secondary", key=f"{adapter.id}_remove", use_container_width=True)

        if remove:
            fts.RemoveAdapter(
                RemoveAdapterRequest(
                    id=adapter.id
                )
            )


display_header()
display_models_section()
