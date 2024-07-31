import streamlit as st
from ft.state import get_state
from ft.app import get_app
from ft.adapter import *
from typing import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random
import os
import torch
from ft.utils import get_device
from ft.utils import attempt_hf_login

# Attempt to log in to huggingface for gated models
hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
attempt_hf_login(hf_token)

# TODO: fix this:
# /Users/jev/miniconda3/envs/jev/lib/python3.10/site-packages/pydantic/_internal/_fields.py:161: UserWarning: Field "model_id" has conflict with protected namespace "model_".

with st.container(border=True):
    col1, col2 = st.columns([1, 17])
    with col1:
        col1.image("./resources/images/difference_24dp_EA3323_FILL0_wght400_GRAD0_opsz40.png")
    with col2:
        col2.subheader('Local Adapter Comparison', divider='red')
        st.caption("Compare your fine-tuned model performance with base model and gain valuable insights.")

st.write("\n")

with st.container(border=True):

    CURRENT_MODEL = None

    current_models = get_state().models
    model_idx = st.selectbox(
        "Base Models",
        range(len(current_models)),
        format_func=lambda x: current_models[x].name,
        index=None
    )

    model_adapter_idx = None

    # TODO: this currently assumes HF model for local eval, but should not have to be in the future
    if model_idx is not None:
        current_model_metadata = current_models[model_idx]

        with st.spinner("Loading model..."):
            CURRENT_MODEL = AutoModelForCausalLM.from_pretrained(
                current_model_metadata.huggingface_model_name, return_dict=True).to(get_device())

        model_adapters: List[AdapterMetadata] = get_state().adapters
        model_adapters = list(filter(lambda x: x.model_id == current_model_metadata.id, model_adapters))

        # Filter adapters based on their presence in the /data/adapter directory
        model_adapters = list(filter(lambda x: os.path.isdir(os.path.join(x.location)), model_adapters))

        # TODO: We should not have to load the adapters every run, this is overkill
        with st.spinner("Loading Adapters..."):
            for adapter in model_adapters:
                loc = os.path.join(adapter.location)
                if not loc.endswith("/"):
                    loc += "/"

                if os.path.isdir(loc):
                    # This is a PEFT Model, we can load another adapter
                    if hasattr(CURRENT_MODEL, 'load_adapter'):
                        CURRENT_MODEL.load_adapter(loc, adapter_name=adapter.id)
                    # This is a regular AutoModelForCausalLM, we should use
                    # PeftModel.from_pretrained for this first adapter load
                    else:
                        raise ValueError("Not supported!")

        model_adapter_idx = st.selectbox(
            "Choose an Adapter",
            range(len(model_adapters)),
            format_func=lambda x: model_adapters[x].name,
            index=None
        )

        if model_adapter_idx is not None:
            model_adapter = model_adapters[model_adapter_idx]


def update_text_area():
    prompts = get_state().prompts
    prompt_idx = st.session_state.input_prompt_idx

    if prompt_idx is not None:
        st.session_state.input_prompt_template = prompts[prompt_idx].prompt_template
    else:
        st.session_state.input_prompt_template = None

# TODO: extend this out to both prompt templates and actual input prompts.
# The button should generate the prompt, the dropdown should allow you to set a prompt template


def generate_random():
    prompts = get_state().prompts
    prompt_template = st.session_state.input_prompt_template
    prompt_idx = st.session_state.input_prompt_idx

    if prompt_template is not None and prompt_idx is not None:
        dataset = load_dataset(get_app().datasets.get_dataset(prompts[prompt_idx].dataset_id).huggingface_name)
        if "train" in dataset:
            dataset = dataset["train"]
        idx = random.randint(0, len(dataset) - 1)
        prompt_string = prompt_template.format(**dataset[idx])
        st.session_state.input_prompt = prompt_string


@st.fragment
def prompt_fragment():
    cont = st.container()

    prompts = get_state().prompts

    prompt_idx = cont.selectbox(
        "Import Prompt Template",
        range(
            len(prompts)),
        key="input_prompt_idx",
        format_func=lambda x: f"{prompts[x].name} [dataset: {get_app().datasets.get_dataset(prompts[x].dataset_id).name}]",
        index=None,
        on_change=update_text_area)
    cont.text_area("Prompt Template", height=120, key="input_prompt_template")

    gen_random = cont.button("Generate Random Prompt from Dataset and Template",
                             on_click=generate_random, use_container_width=True)
    cont.text_area("Input Prompt", height=120, key="input_prompt")


if model_idx is not None and model_adapter_idx is not None:
    prompt_fragment()


@st.fragment
def evaluate_fragment():
    cont = st.container()
    expander = cont.expander("Advanced options")
    expander.caption("TODO: add generation arguments")
    generate_button = cont.button("Generate", type="primary", use_container_width=True)

    if generate_button:
        with st.spinner("Generating text..."):
            tokenizer = AutoTokenizer.from_pretrained(current_model_metadata.huggingface_model_name)

        # input_tokens = tokenizer(st.session_state.input_prompt, return_tensors="pt").to("cpu")
        input_tokens = tokenizer(st.session_state.input_prompt, return_tensors="pt").to(get_device())

        CURRENT_MODEL.disable_adapters()

        with torch.cuda.amp.autocast():
            model_out = CURRENT_MODEL.generate(
                **input_tokens,
                max_new_tokens=50,
                repetition_penalty=1.1,
                num_beams=1,
                temperature=0.7,
                top_p=1.0,
                top_k=50,
                do_sample=True,
            )

        tok_out = tokenizer.decode(model_out[0], skip_special_tokens=False)[len(st.session_state.input_prompt):]
        print(tok_out)

        CURRENT_MODEL.enable_adapters()
        CURRENT_MODEL.set_adapter(model_adapter.id)
        with torch.cuda.amp.autocast():
            model_out = CURRENT_MODEL.generate(
                **input_tokens,
                max_new_tokens=50,
                repetition_penalty=1.1,
                num_beams=1,
                temperature=0.7,
                top_p=1.0,
                top_k=50,
                do_sample=True,
            )
        tok_out2 = tokenizer.decode(model_out[0], skip_special_tokens=False)[len(st.session_state.input_prompt):]
        print(tok_out2)

        st.session_state.base_output = tok_out
        st.session_state.base_output2 = tok_out2

        cont.text(f"Base [model: {current_model_metadata.name}]")
        cont.text_area(f"Base [model: {current_model_metadata.name}]", disabled=True,
                       key="base_output", label_visibility="collapsed", height=200)
        cont.text(f"Base+Adapter [adapter: {model_adapters[model_adapter_idx].name}]")
        cont.text_area(
            f"Base+Adapter [adapter: {model_adapters[model_adapter_idx].name}]",
            key="base_output2",
            disabled=True,
            label_visibility="collapsed",
            height=200)


if model_idx is not None and model_adapter_idx is not None:
    evaluate_fragment()
