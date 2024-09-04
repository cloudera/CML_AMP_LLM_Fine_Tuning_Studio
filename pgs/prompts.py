import streamlit as st
from ft.api import *
from datasets import load_dataset
import random
from uuid import uuid4
from pgs.streamlit_utils import get_fine_tuning_studio_client
import json

from ft.utils import generate_templates

from ft.consts import IconPaths, DIVIDER_COLOR

# Instantiate the client to the FTS gRPC app server.
fts = get_fine_tuning_studio_client()


def display_header():
    with st.container(border=True):
        col1, col2 = st.columns([1, 17])
        with col1:
            col1.image(IconPaths.AIToolkit.CREATE_PROMPTS)
        with col2:
            col2.subheader('Create Prompts', divider=DIVIDER_COLOR)
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
                range(len(datasets)),
                format_func=lambda x: datasets[x].name,
                index=0)  # Set the default index as needed

            if dataset_idx is not None:
                dataset = datasets[dataset_idx]
                st.code("Dataset Columns: \n * " + '\n * '.join(json.loads(dataset.features)))

                columns = json.loads(dataset.features)
                default_prompt_template, default_completion_template = generate_templates(columns)
                subcol1, subcol2 = st.columns(2)
                prompt_template = subcol1.text_area("Prompt Template", value=default_prompt_template, height=260)
                completion_template = subcol2.text_area(
                    "Completion Template", value=default_completion_template, height=260)

                training_prompt_template = prompt_template
                # Remove the newline from the end of the prompt template and concatenate with the response template
                if prompt_template.endswith("\n"):
                    training_prompt_template = prompt_template.rstrip("\n")

                # Concatenate the response within the template properly
                training_prompt_template = training_prompt_template + completion_template.strip()

                generate_example_button = st.button(
                    "Generate Prompt Example", type="secondary", use_container_width=True)

                subcol1, subcol2 = st.columns(2)
                example_training_prompt, example_input_prompt, example_completion_prompt = "", "", ""
                if generate_example_button:
                    with st.spinner("Generating Prompt..."):
                        loaded_dataset = load_dataset(dataset.huggingface_name)

                    dataset_size = len(loaded_dataset["train"])
                    idx_random = random.randint(0, dataset_size - 1)
                    dataset_idx = loaded_dataset["train"][idx_random]

                    # Generate the example prompt and completion using the templates
                    example_training_prompt = training_prompt_template.format(**dataset_idx)
                    example_input_prompt = prompt_template.format(**dataset_idx)
                    example_completion_prompt = completion_template.format(**dataset_idx)

                # Display the example input prompt and completion prompt
                subcol1.caption("Example Training Prompt")
                subcol1.code(example_training_prompt)

                subcol2.caption("Example Prompt")
                subcol2.code(example_input_prompt)

                subcol2.caption("Example Completion")
                subcol2.code(example_completion_prompt)

                if st.button("Create Prompt", type="primary", use_container_width=True):
                    if not new_prompt_name:
                        st.error("Prompt Name cannot be empty!", icon=":material/error:")
                    else:
                        try:
                            add_prompt(
                                new_prompt_name,
                                dataset.id,
                                training_prompt_template,
                                prompt_template,
                                completion_template)
                            st.success("Prompt Created. Please go to **View Prompts** tab.", icon=":material/check:")
                            st.toast("Prompt has been created successfully.", icon=":material/check:")
                        except Exception as e:
                            st.error(f"Failed to create prompt: **{str(e)}**", icon=":material/error:")
                            st.toast(f"Failed to create prompt: **{str(e)}**", icon=":material/error:")

    with col2:
        st.info(
            """
        ### How to Create and Customize Training Prompts

        1. **Prompt Template**
            - **Default Template:** Auto-generated template based on dataset features.

                ```
                You are an LLM. Provide a response below.

                <Instruction>: {instruction}
                <Input>: {input}
                <Response>:
                ```

            - **Modifying Prompt:** Adjust by removing fields or adding instructions.

                ```
                You are a customer chatbot. Please respond politely with information to help the customer based on the intent retrieved.

                <Instruction>: {instruction}
                <Input>: {input}
                <Assistant>:
                ```

        2. **Completion Template**
            - **Default Template:** Defines expected output from the model.

                ```
                {response}
                ```

            - **Customizing Completion:** Add fields for more specific outputs.

                ```
                {response}
                <Source>: {source}
                ```

        3. **Generating and Saving Prompts**
            - **Generate Example:** Preview Input and Completion Prompts with dataset data.
            - **Create Prompt:** Save when satisfied with the templates.
        """
        )


def add_prompt(name, dataset_id, training_prompt_template, prompt_template, completion_template):
    fts.AddPrompt(
        AddPromptRequest(
            prompt=PromptMetadata(
                id=str(uuid4()),
                name=name,
                dataset_id=dataset_id,
                prompt_template=training_prompt_template,
                input_template=prompt_template,
                completion_template=completion_template
            )
        )
    )


display_header()

display_create_prompt()
