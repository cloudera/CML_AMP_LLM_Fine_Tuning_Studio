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
            ["**Import Huggingface Dataset**", "**Import CSV Dataset**", "**Import JSON/JSONL Dataset**"])

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
    c1, c2 = st.columns([2, 4])
    import_dataset_name = c1.text_input(
        'Dataset Name',
        placeholder="My dataset",
        key="csv_import_name")
    import_dataset_location = c2.text_input(
        'CSV Location',
        placeholder="path/to/dataset.csv")
    c3 = st.container()
    with c3:
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
    c1, c2 = st.columns([2, 4])
    import_dataset_name = c1.text_input(
        'Dataset Name',
        placeholder="My dataset",
        key="json_import_name")
    import_dataset_location = c2.text_input(
        'JSON/JSONL File Location',
        placeholder="path/to/dataset.json or dataset.jsonl")
    c3 = st.container()
    with c3:
        import_dataset = c3.button("Import", type="primary", use_container_width=True, key="json_import_button")
    if import_dataset:
        with st.spinner("Loading Dataset..."):
            if import_dataset_name and import_dataset_location:
                try:
                    if import_dataset_location.endswith('.json'):
                        dataset_type = DatasetType.PROJECT_JSON
                    elif import_dataset_location.endswith('.jsonl'):
                        dataset_type = DatasetType.PROJECT_JSONL
                    else:
                        st.error("Unsupported file format! Please upload a `.json` or `.jsonl` file.")
                        return
                    fts.AddDataset(
                        AddDatasetRequest(
                            type=dataset_type,
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
                st.error("Dataset name and location cannot be empty!")

    st.write("\n")
    st.info("""
        **JSON and JSON Lines Dataset**
        The Fine Tuning Studio allows users to import custom JSON or JSON Lines (JSONL) datasets from a project file.
        **How to Prepare Data for Import:**
        Studio supports importing datasets in either **JSON** or **JSON Lines (JSONL)** format. The file must have the following structure:

        **For JSON Files (`*.json`):**
        - The file should contain a single JSON object or an array of JSON objects (list of dictionaries).
        - Each dictionary represents a dataset entry, with keys representing feature names. Feature names must not contain whitespace.
        Example:
        ```json
        [
            {"feature1": "value1", "feature2": "value2"},
            {"feature1": "value3", "feature2": "value4"}
        ]
        ```

        **For JSON Lines Files (`*.jsonl`):**
        - The file should contain one JSON object per line, with each object representing a dataset entry.
        - Each object must follow the same structure, and feature names must not contain whitespace.
        Example:
        ```json
        {"feature1": "value1", "feature2": "value2"}
        {"feature1": "value3", "feature2": "value4"}
        ```

        **Data Consistency:** All entries are expected to be consistent and can be converted into a dataset format supported by the `datasets` directory.

        **Single Split Datasets Only:** Currently, only "single split" datasets are supported. You cannot import separate train and test splits within the same file.

        **How to Import:**
        - Enter the dataset name in the input field.
        - Enter the file location in the location field.
        - Click the "Import" button.
        - The dataset will be loaded and can be used for training an adapter on a foundational model.

        **Usage:**
        Once imported, these datasets can be utilized to:
        - fine-tune adapters on foundational models,
        - create prompts for these datasets, and
        - run evaluations against the dataset.
    """, icon=":material/info:")
    return


create_header()
handle_database_import()
