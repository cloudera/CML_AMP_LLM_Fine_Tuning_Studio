import streamlit as st
from ft.api import *
from datasets import load_dataset
import random
from uuid import uuid4
from pgs.streamlit_utils import get_fine_tuning_studio_client
import json

# Instantiate the client to the FTS gRPC app server.
fts = get_fine_tuning_studio_client()


def display_header():
    with st.container(border=True):
        col1, col2 = st.columns([1, 17])
        with col1:
            col1.image("./resources/images/chat_24dp_EA3323_FILL0_wght400_GRAD0_opsz48.png")
        with col2:
            col2.subheader('Create Prompts')
            col2.caption(
                'Generate tailored prompts for your fine-tuning tasks on the specified datasets and models to enhance performance.')


def display_create_prompt():
    loaded_dataset = None

    col1, col2 = st.columns([3, 2])
    with col1:
        with st.container(border=True):
            new_prompt_name = st.text_input("Prompt Name", placeholder="Enter a human-friendly prompt name")

            datasets = fts.get_datasets()
            dataset_idx = st.selectbox(
                "Dataset",
                range(
                    len(datasets)),
                format_func=lambda x: datasets[x].name,
                index=0)  # Set the default index as needed

            if dataset_idx is not None:
                dataset = datasets[dataset_idx]
                st.code("Dataset Columns: \n * " + '\n * '.join(json.loads(dataset.features)))

                default_template = ""
                for feature in json.loads(dataset.features):
                    default_template += f"<{feature.capitalize()}>: {{{feature}}}\n"
                prompt_template = st.text_area("Prompt Template", value=default_template, height=200)

                generate_example_button = st.button(
                    "Generate Prompt Example", type="secondary", use_container_width=True)

                prompt_output = ""
                if generate_example_button:
                    with st.spinner("Generating Prompt..."):
                        loaded_dataset = load_dataset(dataset.huggingface_name)

                    dataset_size = len(loaded_dataset["train"])
                    idx_random = random.randint(0, dataset_size - 1)
                    dataset_idx = loaded_dataset["train"][idx_random]
                    prompt_output = prompt_template.format(**dataset_idx)

                st.caption("**Example Prompt**")
                st.code(prompt_output)

                if st.button("Create Prompt", type="primary", use_container_width=True):
                    if not new_prompt_name:
                        st.error("Prompt Name cannot be empty!", icon=":material/error:")
                    else:
                        try:
                            add_prompt(new_prompt_name, dataset.id, prompt_template)
                            st.success("Prompt Created. Please go to **View Prompts** tab.", icon=":material/check:")
                            st.toast("Prompt has been created successfully.", icon=":material/check:")
                        except Exception as e:
                            st.error(f"Failed to create prompt: **{str(e)}**", icon=":material/error:")
                            st.toast(f"Failed to create prompt: **{str(e)}**", icon=":material/error:")

    with col2:
        st.info("""
        ### How to Create and Customize Training Prompts

        Creating effective prompts is key to fine-tuning models. Follow these steps:

        ### 1. Customizing the Prompt Template
        - **Default Template**: The text box will auto-generate a prompt template based on dataset features. For example:
        ```
        <Instruction>: {instruction}
        <Input>: {input}
        <Response>: {response}
        <Text>: {text}
        ```
        - **Modifying Prompt**: Remove irrelevant fields and add context to better suit your task. For instance:
        ```
        Write the response as an SQL query.
        <Instruction>: {instruction}
        <Input>: {input}
        <Response>: {response}
        ```

        ### 2. Generating and Saving Prompts
        - Click **Generate Prompt Example** to see your template in action and adjust as needed and **Save** the prompt.
        - Manage your prompts in the **Available Prompts** tab.

        Use these steps to effectively create and manage training prompts!
        """)


def add_prompt(name, dataset_id, template):
    fts.AddPrompt(
        AddPromptRequest(
            prompt=PromptMetadata(
                id=str(uuid4()),
                name=name,
                dataset_id=dataset_id,
                prompt_template=template
            )
        )
    )


display_header()

display_create_prompt()
