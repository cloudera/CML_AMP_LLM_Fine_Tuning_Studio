import streamlit as st
from ft.dataset import DatasetMetadata, DatasetType
from ft.app import get_app
from ft.consts import HF_LOGO
from typing import List

def display_header():
    with st.container(border=True):
        col1, col2 = st.columns([1, 13])
        with col1:
            col1.image("./resources/images/data_object_24dp_EA3323_FILL0_wght400_GRAD0_opsz48.png")
        with col2:
            col2.subheader('Available Datasets', divider='orange')
            st.write("Explore and organize imported datasets from Hugging Face or custom sources. Gain insights into the structure and content of each dataset.")

def display_datasets(datasets: List[DatasetMetadata], dataset_type: DatasetType, logo_path: str, no_dataset_message: str):
    filtered_datasets = [dataset for dataset in datasets if dataset.type == dataset_type]
    
    if not filtered_datasets:
        st.info(no_dataset_message)
    else:
        for i, dataset in enumerate(filtered_datasets):
            if i % 2 == 0:
                col1, col2 = st.columns(2)
            
            ds_cont = col1 if i % 2 == 0 else col2
            
            with ds_cont.container(border=True, height=200):
                c1, c2, c2empty, c3 = st.columns([2, 9, 1, 4])
                c1.image(logo_path, use_column_width=True)
                c2.markdown(f"**{dataset.name}**")
                if dataset.description:
                    c2.text(dataset.description)
                c2.code("Features: \n * " + '\n * '.join(dataset.features))

                remove = c3.button("Remove", key=f"{dataset.id}_remove", type="primary", use_container_width=True)

                if remove:
                    get_app().remove_dataset(dataset.id)
                    st.rerun()

display_header()
st.write("\n")

datasets: List[DatasetMetadata] = get_app().datasets.list_datasets()

tab1, tab2 = st.tabs(["**Huggingface Datasets**", "**Custom Datasets**"])

with tab1:
    display_datasets(datasets, DatasetType.HUGGINGFACE, "./resources/images/hf-logo.png", "No Huggingface datasets available.")

with tab2:
    display_datasets(datasets, DatasetType.IMPORTED, HF_LOGO, "No custom datasets available.")


