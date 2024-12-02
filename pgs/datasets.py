import streamlit as st
from ft.api import *
from pgs.streamlit_utils import get_fine_tuning_studio_client
from ft.consts import IconPaths, DIVIDER_COLOR

# Instantiate the client to the FTS gRPC app server.
fts = get_fine_tuning_studio_client()


def create_header():
    with st.container(border=True):
        col1, col2 = st.columns([1, 17])
        with col1:
            col1.image(IconPaths.AIToolkit.IMPORT_DATASETS)
        with col2:
            col2.subheader('Import Datasets', divider=DIVIDER_COLOR)
            st.caption(
                "Import the datasets from either available datasets on Hugging Face or upload your own preprocessed dataset from local.")


def handle_database_import():
    with st.container():
        import_hf_tab, csv_tab, json_tab = st.tabs(
            ["**Import Huggingface Dataset**", "**Import CSV Dataset**", "**Import JSON Dataset**"])

        with import_hf_tab:
            display_huggingface_import_tab()
        with csv_tab:
            display_csv_import_tab()
        with json_tab:
            display_json_import_tab()


def display_huggingface_import_tab():
    c1, c2 = st.columns([4, 1])
    import_hf_dataset_name = c1.text_input(
        'Dataset Name',
        placeholder="organization/dataset",
        label_visibility='collapsed')
    import_hf_dataset = c2.button("Import", type="primary", use_container_width=True, key="hf_import")

    if import_hf_dataset:
        with st.spinner("Loading Dataset..."):
            if import_hf_dataset_name:
                try:
                    fts.AddDataset(
                        AddDatasetRequest(
                            type=DatasetType.HUGGINGFACE,
                            huggingface_name=import_hf_dataset_name,
                            location=None
                        )
                    )
                    st.success("Dataset Loaded. Please go to **View Dataset** tab.", icon=":material/check:")
                    st.toast("Dataset has been loaded successfully.", icon=":material/check:")
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
    return


def display_csv_import_tab():
    c1, c2, c3 = st.columns([2, 4, 1])
    import_dataset_name = c1.text_input(
        'Dataset Name',
        placeholder="My dataset",
        key="csv_import_name")
    import_dataset_location = c2.text_input(
        'CSV Location',
        placeholder="path/to/dataset.csv")
    import_dataset = c3.button("Import", type="primary", use_container_width=True, key="csv_import_location")

    if import_dataset:
        with st.spinner("Loading Dataset..."):
            if import_dataset_name and import_dataset_location:
                try:
                    fts.AddDataset(
                        AddDatasetRequest(
                            type=DatasetType.PROJECT_CSV,
                            name=import_dataset_name,
                            location=import_dataset_location
                        )
                    )
                    st.success("Dataset Loaded. Please go to **View Dataset** tab.", icon=":material/check:")
                    st.toast("Dataset has been loaded successfully.", icon=":material/check:")
                except Exception as e:
                    st.error(f"Failed to load dataset: **{str(e)}**", icon=":material/error:")
                    st.toast(f"Failed to load dataset: **{str(e)}**", icon=":material/error:")
            else:
                st.error("Dataset name cannot be empty!")

    st.write("\n")
    st.info("""
        **CSV Dataset**

        The Fine Tuning Studio allows for users to import custom CSV datasets from a project file.

        **How to Prepare Data for Import:**
        - Studio currently supports importing a singular dataset CSV file that ends in *.csv.
        - The first row of the CSV represents the dataset feature names, and these features should not have any whitespace in the names.
        - All data is assumed to be consistent (able to be cast into a `Dataset` format from the `datasets` directory)
        - Right now only "single split" datasets are supported. (i.e., you cannot import a train *and* test split with one dataset file).

        **How to Import:**
        - Enter the dataset name in the input field.
        - Enter the dataset CSV location in the location field.
        - Click the "Import" button.
        - The dataset will be loaded and can be used for training an adapter on a foundational model.

        **Usage:**
        - Once imported, these datasets can be utilized to:
          - fine-tune adapters on foundational models,
          - create prompts for these datasets, and
          - run evaluations against the dataset.

    """, icon=":material/info:")
    return


def display_json_import_tab():
    c1, c2, c3 = st.columns([2, 4, 1])
    import_dataset_name = c1.text_input(
        'Dataset Name',
        placeholder="My dataset",
        key="json_import_name")
    import_dataset_location = c2.text_input(
        'JSON File Location',
        placeholder="path/to/dataset.json")
    import_dataset = c3.button("Import", type="primary", use_container_width=True, key="json_import_location")

    if import_dataset:
        with st.spinner("Loading Dataset..."):
            if import_dataset_name and import_dataset_location:
                try:
                    fts.AddDataset(
                        AddDatasetRequest(
                            type=DatasetType.PROJECT_JSON,
                            name=import_dataset_name,
                            location=import_dataset_location
                        )
                    )
                    st.success("Dataset Loaded. Please go to **View Dataset** tab.", icon=":material/check:")
                    st.toast("Dataset has been loaded successfully.", icon=":material/check:")
                except Exception as e:
                    st.error(f"Failed to load dataset: **{str(e)}**", icon=":material/error:")
                    st.toast(f"Failed to load dataset: **{str(e)}**", icon=":material/error:")
            else:
                st.error("Dataset name cannot be empty!")

    st.write("\n")
    st.info("""
        **JSON Dataset**

        The Fine Tuning Studio allows for users to import custom JSON datasets from a project file.

        **How to Prepare Data for Import:**
        - Studio currently supports importing a singular dataset JSON file that ends in *.json.
        - The JSON file format should be a singular list of dicts.
        - Each dict key represents a dataset feature name, and these features should not have any whitespace in the names.
        - All data is assumed to be consistent (able to be cast into a `Dataset` format from the `datasets` directory)
        - Right now only "single split" datasets are supported. (i.e., you cannot import a train *and* test split with one dataset file).

        **How to Import:**
        - Enter the dataset name in the input field.
        - Enter the dataset JSON location in the location field.
        - Click the "Import" button.
        - The dataset will be loaded and can be used for training an adapter on a foundational model.

        **Usage:**
        - Once imported, these datasets can be utilized to:
          - fine-tune adapters on foundational models,
          - create prompts for these datasets, and
          - run evaluations against the dataset.

    """, icon=":material/info:")
    return


create_header()
handle_database_import()
