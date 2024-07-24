import streamlit as st
from ft.dataset import DatasetMetadata
from ft.app import get_app
from ft.dataset import *
from ft.consts import HF_LOGO

st.subheader("Available Datasets")

datasets: List[DatasetMetadata] = get_app().datasets.list_datasets()

tab1, tab2 = st.tabs(["Huggingface Datasets", "Custom Datasets"])

with tab1:
    cont = st.container(border=False)
    huggingface_datasets = [dataset for dataset in datasets if dataset.type == DatasetType.HUGGINGFACE]
    
    if not huggingface_datasets:
        st.info("No Huggingface datasets available.")
    else:
        for i, dataset in enumerate(huggingface_datasets):
            if i % 2 == 0:
                col1, col2 = cont.columns(2)
            
            ds_cont = col1 if i % 2 == 0 else col2
            
            with ds_cont.container(border=True, height=200):
                c1, c2, c3 = st.columns([1, 4, 1])
                c1.image("./resources/images/hf-logo.svg", use_column_width=True)
                c2.markdown(f"**{dataset.name}**")
                if dataset.description:
                    c2.text(dataset.description)
                c2.text("Features: \n * " + '\n * '.join(dataset.features))

                remove = c3.button("Remove", key=f"{dataset.id}_remove", type="primary")

                if remove:
                    get_app().remove_dataset(dataset.id)
                    st.rerun()

with tab2:
    custom_datasets = [dataset for dataset in datasets if dataset.type != DatasetType.HUGGINGFACE]
    
    if not custom_datasets:
        tab2.info("No custom datasets available.")
    else:
        for i, dataset in enumerate(custom_datasets):
            if i % 2 == 0:
                col1, col2 = cont.columns(2)
            
            ds_cont = col1 if i % 2 == 0 else col2
            
            with ds_cont.container(border=True, height=200):
                c1, c2, c3 = st.columns([1, 4, 1])
                c1.image(HF_LOGO, use_column_width=True)
                c2.markdown(f"**{dataset.name}**")
                if dataset.description:
                    c2.text(dataset.description)
                c2.text("Features: \n * " + '\n * '.join(dataset.features))

                remove = c3.button("Remove", key=f"{dataset.id}_remove", type="primary")

                if remove:
                    get_app().remove_dataset(dataset.id)
                    st.rerun()
