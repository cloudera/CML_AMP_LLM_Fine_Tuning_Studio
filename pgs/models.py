import streamlit as st

st.title("Base Models")

base_models_container = st.container(height=500, border=False)

base_models = [
    {
        "name": "bigscience/bloom-1b1",
        "type": "huggingface",
        "huggingface": {
            "model": "bigscience/bloom-1b1",
            "url": "https://huggingface.co/bigscience/bloom-1b1"
        } 
    },
    {
        "name": "bigscience/bloom-1b1",
        "type": "huggingface",
        "huggingface": {
            "model": "bigscience/bloom-1b1",
            "url": "https://huggingface.co/bigscience/bloom-1b1"
        } 
    },
    {
        "name": "bigscience/bloom-1b1",
        "type": "huggingface",
        "huggingface": {
            "model": "bigscience/bloom-1b1",
            "url": "https://huggingface.co/bigscience/bloom-1b1"
        } 
    },
    {
        "name": "bigscience/bloom-1b1",
        "type": "huggingface",
        "huggingface": {
            "model": "bigscience/bloom-1b1",
            "url": "https://huggingface.co/bigscience/bloom-1b1"
        } 
    }
]


for base_model in base_models:
    model_container = base_models_container.expander(base_model["name"])
    model_container.header(base_model["name"])
    model_container.caption("some caption")
    model_container.link_button("Find on Huggingface", base_model["huggingface"]["url"])





