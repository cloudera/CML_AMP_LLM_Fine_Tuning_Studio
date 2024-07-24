import streamlit as st
from ft.dataset import DatasetMetadata
from ft.app import get_app
from ft.dataset import *
from ft.consts import HF_LOGO
from ft.state import get_state
from datasets import load_dataset
import random
from uuid import uuid4
from ft.prompt import PromptMetadata

st.subheader("Create Prompts")

loaded_dataset = None

with st.container(border=True):
    new_prompt_name = st.text_input("Prompt Name", placeholder="Enter a human-friendly prompt name")

    # dataset = st.selectbox("Dataset", [f"{X.name}" for X in get_state().datasets], index=None)

    datasets = get_state().datasets
    dataset_idx = st.selectbox("Dataset", range(len(datasets)), format_func=lambda x: datasets[x].name, index=None)
    

    if dataset_idx is not None:
        dataset = datasets[dataset_idx]

        st.text("Dataset Columns: \n * " + '\n * '.join(dataset.features))

        c1, c2 = st.columns([1, 1])
        prompt_template_header = c1.subheader("Enter Prompt Template")
        generate_example_button = c2.button("Generate Prompt Example", type="primary", use_container_width=True)

        cc1, cc2 = st.columns([1, 1])
        prompt_template = cc1.text_area("Prompt Template", height=300, label_visibility="collapsed")

        prompt_output = None 
        if generate_example_button:
            with st.spinner("Loading Dataset..."):

                # Assuming that the HF dataset will be cached from here
                loaded_dataset = load_dataset(dataset.huggingface_name)

            dataset_size = len(loaded_dataset["train"])
            idx_random = random.randint(0, dataset_size-1)

            dataset_idx = loaded_dataset["train"][idx_random]
            prompt_output = prompt_template.format(**dataset_idx)
                
        cc2.text_area("Example Prompt", value=prompt_output, height=300, disabled=True, label_visibility="collapsed")

        create_prompt = st.button("Create Prompt", type="primary")

        if create_prompt:
            get_app().add_prompt(PromptMetadata(
                id=str(uuid4()),
                name=new_prompt_name,
                dataset_id=dataset.id,
                slots=None,
                prompt_template=prompt_template
            ))


st.markdown("---")
st.subheader("Available Prompts")

prompts: List[PromptMetadata] = get_state().prompts

cont = st.container(border=False)

for prompt in prompts:
    p_c = cont.container(border=True)
    c1, c2 = p_c.columns([4,1])
    c1.markdown(f"**{prompt.name}**")
    c1.caption(prompt.id)
    remove = c2.button("Remove", type="primary", key=f"{prompt.id}_remove_button", use_container_width=True)

    c1, c2 = p_c.columns([4,1])
    c1.text_area("Template", key=f"{prompt.id}_prompt_template", value=prompt.prompt_template, height=100, disabled=True, label_visibility="collapsed")

    c2.text("Dataset:")
    c2.caption(get_app().datasets.get_dataset(prompt.dataset_id).name)

    if remove:
        get_app().remove_prompt(prompt.id)
        st.rerun()
