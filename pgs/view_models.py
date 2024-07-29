import streamlit as st
from ft.state import get_state
from ft.app import get_app
from ft.model import ModelMetadata, ModelType, ImportModelRequest
from ft.adapter import AdapterMetadata, AdapterType
from typing import List


def display_header():
    with st.container(border=True):
        col1, col2 = st.columns([1, 17])
        with col1:
            col1.image("./resources/images/view_day_24dp_EA3323_FILL0_wght400_GRAD0_opsz48.png")
        with col2:
            col2.subheader('Base Models & Adapters', divider='red')
            st.caption("Review the imported base models and the adapters generated during the fine-tuning process. Note that a single model can contain multiple adapters.")


def display_models_section():
    models: List[ModelMetadata] = get_state().models
    adapters: List[AdapterMetadata] = get_state().adapters

    with st.container():
        tab1, tab2 = st.tabs(["**Huggingface Models**", "**Local Models**"])
        with tab1:
            display_models([model for model in models if model.type == ModelType.HUGGINGFACE], adapters)
        with tab2:
            display_models([model for model in models if model.type != ModelType.HUGGINGFACE], adapters)


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
                get_app().remove_model(model.id)
                st.rerun()

            model_adapters = [adapter for adapter in adapters if adapter.model_id == model.id]
            expander = ds_cont.expander("Adapters")

            if not model_adapters:
                expander.info(
                    "No available adpater to display. Please go to **Train a new Adapater** page.",
                    icon=":material/info:")
            for adapter in model_adapters:
                display_adapter(adapter, expander)

            # add_adapter_button = expander.button("Add Adapter", type="primary", key=f"{model.id}_add_adapter", use_container_width=True)
            # if add_adapter_button:
            #     st.toast("You can't do that yet.", icon=":material/info:")


def display_adapter(adapter: AdapterMetadata, container):
    with container:
        c1, c2 = container.columns([4, 1])
        c1.text(adapter.name)
        if adapter.type == AdapterType.LOCAL:
            c1.caption(adapter.location)

        remove = c2.button("Remove", type="secondary", key=f"{adapter.id}_remove", use_container_width=True)

        if remove:
            st.toast("You can't do that yet.", icon=":material/info:")


display_header()

display_models_section()
