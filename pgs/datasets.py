import streamlit as st
from ft.dataset import DatasetMetadata, ImportDatasetRequest, DatasetType
from ft.app import get_app
from ft.consts import HF_LOGO

def create_header():
    with st.container(border=True):
        col1, col2 = st.columns([1, 13])
        with col1:
            col1.image("./resources/images/publish_24dp_EA3323_FILL0_wght400_GRAD0_opsz48.png")
        with col2:
            col2.subheader('Import Datasets', divider='orange')
            st.write("Import the datasets from either available datasets on Hugging Face or upload your own preprocessed dataset from local.")

def handle_database_import():
    with st.container():
        import_hf_tab, upload_tab = st.tabs(["**Import Huggingface Dataset**", "**Upload Custom Dataset**"])

        with import_hf_tab:
            c1, c2 = st.columns([4, 1])
            import_hf_dataset_name = c1.text_input('Dataset Name', placeholder="huggingface/dataset", label_visibility='collapsed')
            import_hf_dataset = c2.button("Import", type="primary", use_container_width=True)

            if import_hf_dataset:
                with st.spinner("Loading Dataset..."):
                    if import_hf_dataset_name:
                        try:
                            get_app().add_dataset(ImportDatasetRequest(type=DatasetType.HUGGINGFACE, huggingface_name=import_hf_dataset_name, location=None))
                            st.success("Dataset Loaded!")
                        except Exception as e:
                            st.error(f"Failed to load dataset: {str(e)}")
                    else:
                        st.error("Dataset name cannot be empty!")
        with upload_tab:
            with st.container():
                data_file = st.file_uploader("Upload a dataset", type=["csv", "txt", "json"], disabled=True)

                if data_file:
                    st.warning("Custom dataset uploads not yet supported.")
                    # Add data handling logic here if needed



create_header()

st.write("\n\n")

handle_database_import()

