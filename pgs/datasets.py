import streamlit as st
from ft.dataset import DatasetMetadata
from ft.app import get_app
from ft.dataset import *
from ft.consts import HF_LOGO

st.title("Datasets")


with st.container(border=True):
    st.header("New Dataset")

    upload_ct = st.container(border=True)
    import_hf_tab, upload_tab = upload_ct.tabs(["Import Huggingface Dataset", "Upload Custom Dataset"])

    import_hf_tab.subheader("Import Huggingface Dataset")
    c1, c2 = import_hf_tab.columns([4, 1])
    import_hf_dataset_name = c1.text_input('Dataset Name', placeholder="huggingface/dataset", label_visibility='collapsed')
    import_hf_dataset = c2.button("Import", use_container_width=True)

    if import_hf_dataset:
        with st.spinner("Loading Dataset..."):
            if import_hf_dataset_name:
                get_app().add_dataset(ImportDatasetRequest(type=DatasetType.HUGGINGFACE, huggingface_name=import_hf_dataset_name, location=None))
                st.info("Dataset loaded!")
            else:
                st.warning("No dataset name added.")

    upload_tab.header("Upload Dataset")
    data_file = upload_tab.file_uploader("Upload a dataset", type=["csv", "txt", "json"])

    if data_file:
        st.warning("Custom dataset uploads not yet supported.")
        # Add data handling logic here


st.header("Available Datasets")

datasets: List[DatasetMetadata] = get_app().datasets.list_datasets()

cont = st.container(border=False)

for dataset in datasets:
    ds_cont = cont.container(border=True)
    
    if dataset.type == DatasetType.HUGGINGFACE:
        c1, c2, c3 = ds_cont.columns([1, 5, 1])
        c1.image(HF_LOGO, dataset.huggingface_name)
        c2.subheader(dataset.name)
        c2.caption(dataset.id)
        if dataset.description:
            c2.text(dataset.description)
        # c2.divider()
        c2.text("Features: \n * " + '\n * '.join(dataset.features))

        remove = c3.button("Remove", key=f"{dataset.id}_remove")

        if remove:
            get_app().remove_dataset(dataset.id)
            st.rerun()
        
