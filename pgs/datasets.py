import streamlit as st
from ft.dataset import DatasetMetadata
from ft.app import get_app
from ft.dataset import *
from ft.consts import HF_LOGO

with st.container():
    st.subheader("Import Datasets")

    upload_ct = st.container(border=True)
    import_hf_tab, upload_tab = upload_ct.tabs(["Import Huggingface Dataset", "Upload Custom Dataset"])

    c1, c2 = import_hf_tab.columns([4, 1])
    import_hf_dataset_name = c1.text_input('Dataset Name', placeholder="huggingface/dataset", label_visibility='collapsed')
    import_hf_dataset = c2.button("Import", type="primary", use_container_width=True)

    if import_hf_dataset:
        with st.spinner("Loading Dataset..."):
            if import_hf_dataset_name:
                get_app().add_dataset(ImportDatasetRequest(type=DatasetType.HUGGINGFACE, huggingface_name=import_hf_dataset_name, location=None))
                st.info("Dataset loaded!")
            else:
                st.warning("No dataset name added.")

    data_file = upload_tab.file_uploader("Upload a dataset", type=["csv", "txt", "json"])

    if data_file:
        st.warning("Custom dataset uploads not yet supported.")
        # Add data handling logic here

