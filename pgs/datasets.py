import streamlit as st
from ft.dataset import DatasetMetadata, ImportDatasetRequest, DatasetType
from ft.app import get_app
from ft.consts import HF_LOGO


def create_header():
    with st.container(border=True):
        col1, col2 = st.columns([1, 17])
        with col1:
            col1.image("./resources/images/publish_24dp_EA3323_FILL0_wght400_GRAD0_opsz48.png")
        with col2:
            col2.subheader('Import Datasets', divider='red')
            st.caption(
                "Import the datasets from either available datasets on Hugging Face or upload your own preprocessed dataset from local.")


def handle_database_import():
    with st.container():
        import_hf_tab = st.tabs(["**Import Huggingface Dataset**"])

        with import_hf_tab[0]:
            c1, c2 = st.columns([4, 1])
            import_hf_dataset_name = c1.text_input(
                'Dataset Name',
                placeholder="organization/dataset",
                label_visibility='collapsed')
            import_hf_dataset = c2.button("Import", type="primary", use_container_width=True)

            if import_hf_dataset:
                with st.spinner("Loading Dataset..."):
                    if import_hf_dataset_name:
                        try:
                            get_app().add_dataset(
                                ImportDatasetRequest(
                                    type=DatasetType.HUGGINGFACE,
                                    huggingface_name=import_hf_dataset_name,
                                    location=None))
                            st.success("Dataset Loaded. Please go to **View Dataset** tab!", icon=":material/check:")
                            st.toast("Dataset has been loaded successfully!", icon=":material/check:")
                        except Exception as e:
                            st.error(f"Failed to load dataset: **{str(e)}**", icon=":material/error:")
                            st.toast(f"Failed to load dataset: **{str(e)}**", icon=":material/error:")
                    else:
                        st.error("Dataset name cannot be empty!")

            st.write("\n")
            st.info("""
                **Hugging Face Datasets:**

                Hugging Face provides a wide range of datasets that can be used for various machine learning tasks. To import a dataset from Hugging Face, you need to specify the dataset name in the format `organization/dataset`.

                **How to Find Organization and Dataset Name:**
                - Visit the Hugging Face Datasets [website](https://huggingface.co/datasets).
                - Browse or search for the dataset you need.
                - The dataset name is shown in the format `organization/dataset` on the dataset's page.

                **How to Import:**
                - Enter the dataset name in the input field.
                - Click the "Import" button.
                - The dataset will be loaded and can be used for training an adapter on a foundational model.

                **Usage:**
                - Once imported, these datasets can be utilized to fine-tune adapters on foundational models, enhancing their performance for specific tasks.

            """, icon=":material/info:")


create_header()
handle_database_import()
