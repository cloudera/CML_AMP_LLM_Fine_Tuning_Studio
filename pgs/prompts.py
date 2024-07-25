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
from typing import List

def display_header():
    with st.container(border=True):
        col1, col2 = st.columns([1, 13])
        with col1:
            col1.image("./resources/images/chat_24dp_EA3323_FILL0_wght400_GRAD0_opsz48.png")
        with col2:
            col2.subheader('Prompts', divider='orange')
            col2.write('Generate tailored prompts for your fine-tuning tasks on the specified datasets and models to enhance performance.')
    st.write("\n")

def display_create_prompt():
    loaded_dataset = None

    with st.container(border=True):
        new_prompt_name = st.text_input("Prompt Name", placeholder="Enter a human-friendly prompt name")

        datasets = get_state().datasets
        dataset_idx = st.selectbox("Dataset", range(len(datasets)), format_func=lambda x: datasets[x].name, index=None)

        if dataset_idx is not None:
            dataset = datasets[dataset_idx]
            st.code("Dataset Columns: \n * " + '\n * '.join(dataset.features))

            c1, c2 = st.columns([1, 1])
            prompt_template_header = c1.button("Enter Prompt Template", use_container_width=True)
            generate_example_button = c2.button("Generate Prompt Example", type="primary", use_container_width=True)

            cc1, cc2 = st.columns([1, 1])
            prompt_template = cc1.text_area("Prompt Template", height=300, label_visibility="collapsed")

            prompt_output = None 
            if generate_example_button:
                with st.spinner("Loading Dataset..."):
                    loaded_dataset = load_dataset(dataset.huggingface_name)

                dataset_size = len(loaded_dataset["train"])
                idx_random = random.randint(0, dataset_size - 1)
                dataset_idx = loaded_dataset["train"][idx_random]
                prompt_output = prompt_template.format(**dataset_idx)

            cc2.text_area("Example Prompt", value=prompt_output, height=300, disabled=True, label_visibility="collapsed")

            create_prompt = st.button("Create Prompt", type="primary", use_container_width=True)

            if create_prompt:
                add_prompt(new_prompt_name, dataset.id, prompt_template)

def add_prompt(name, dataset_id, template):
    get_app().add_prompt(PromptMetadata(
        id=str(uuid4()),
        name=name,
        dataset_id=dataset_id,
        slots=None,
        prompt_template=template
    ))

def display_available_prompts():
    prompts: List[PromptMetadata] = get_state().prompts

    if not prompts:
        st.info("No prompts available.")
        return

    cont = st.container(border=False)

    for prompt in prompts:
        display_prompt(prompt, cont)

def display_prompt(prompt: PromptMetadata, container):
    with container.container(border=True):
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"**{prompt.name}**")
        c1.caption(prompt.id)
        remove = c2.button("Remove", type="primary", key=f"{prompt.id}_remove_button", use_container_width=True)

        c1, c2 = st.columns([4, 1])
        c1.code(prompt.prompt_template)

        c2.text("Dataset:")
        c2.caption(get_app().datasets.get_dataset(prompt.dataset_id).name)

        if remove:
            get_app().remove_prompt(prompt.id)
            st.rerun()

display_header()
create_prompt_tab, available_prompts_tab = st.tabs(["**Create Prompt**", "**Available Prompts**"])

with create_prompt_tab:
    display_create_prompt()

with available_prompts_tab:
    display_available_prompts()