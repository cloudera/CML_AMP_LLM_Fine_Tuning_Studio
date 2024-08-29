import streamlit as st
from ft.api import *
from typing import List
from pgs.streamlit_utils import get_fine_tuning_studio_client

# Instantiate the client to the FTS gRPC app server.
fts = get_fine_tuning_studio_client()


def display_header():
    with st.container(border=True):
        col1, col2 = st.columns([1, 17])
        with col1:
            col1.image("./resources/images/forum_24dp_EA3323_FILL0_wght400_GRAD0_opsz48.png")
        with col2:
            col2.subheader('View Prompts')
            col2.caption(
                'View all the created prompts for your fine-tuning tasks')


def display_available_prompts():
    prompts: List[PromptMetadata] = fts.get_prompts()

    if not prompts:
        st.info("No prompts available.")
        return

    col1, col2 = st.columns(2)

    for i, prompt in enumerate(prompts):
        container = col1 if i % 2 == 0 else col2
        display_prompt(prompt, container)


def display_prompt(prompt: PromptMetadata, container):
    with container.container(height=200):
        c1, c2 = st.columns([3, 1])
        c1.markdown(f"**{prompt.name}**")
        # remove = c2.button("Remove", type="primary", key=f"{prompt.id}_remove_button", use_container_width=True)

        c1, _, c2 = st.columns([30, 2, 10])
        c1.code(prompt.prompt_template)

        c2.text("Dataset:")
        c2.caption(fts.GetDataset(
            GetDatasetRequest(
                id=prompt.dataset_id
            )
        ).dataset.name)

        # if remove:
        #     fts.RemovePrompt(
        #         RemovePromptRequest(
        #             id=prompt.id
        #         )
        #     )
        #     st.rerun()


display_header()

display_available_prompts()
