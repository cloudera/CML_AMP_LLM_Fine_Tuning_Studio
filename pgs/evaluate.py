import streamlit as st
from ft.api import *
from typing import *
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import random
import os
import torch
from ft.utils import get_device
from ft.utils import attempt_hf_login
from pgs.streamlit_utils import get_fine_tuning_studio_client
import json
import time

# Instantiate (or get the pre-existing) client to the FTS gRPC app server.
fts = get_fine_tuning_studio_client()

# Initialize session state
if 'current_model_metadata' not in st.session_state:
    st.session_state.current_model_metadata = None
if 'prv_model_metadata' not in st.session_state:
    st.session_state.prv_model_metadata = None
if 'model_adapters' not in st.session_state:
    st.session_state.model_adapters = []
if 'base_output' not in st.session_state:
    st.session_state.base_output = ""
if 'adapter_outputs' not in st.session_state:
    st.session_state.adapter_outputs = {}
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'loaded_adapters' not in st.session_state:
    st.session_state.loaded_adapters = []
if 'generation_config_text' not in st.session_state:
    st.session_state.generation_config_text = ""
if 'input_prompt' not in st.session_state:
    st.session_state.input_prompt = ""
if 'input_prompt_template' not in st.session_state:
    st.session_state.input_prompt_template = None
if 'input_prompt_idx' not in st.session_state:
    st.session_state.input_prompt_idx = None
if 'selected_adapters' not in st.session_state:
    st.session_state.selected_adapters = []
if 'model_idx'  not in st.session_state:
    st.session_state.model_idx = None

# Handle Huggingface login attempt
hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
if hf_token:
    attempt_hf_login(hf_token)

def on_model_change():
    st.session_state.model_idx = st.session_state.selected_model_idx

def on_adapters_change():
    st.session_state.selected_adapters = st.session_state.selected_adapters_key_name

def on_text_area_change():
    st.session_state.input_prompt_template = st.session_state.text_area_value

def on_input_prompt_change():
    st.session_state.input_prompt = st.session_state.input_prompt_key

def update_text_area():
    st.session_state.input_prompt_idx = st.session_state.input_prompt_idx_key
    try:
        prompts = fts.get_prompts()
        prompt_idx = st.session_state.input_prompt_idx

        if prompt_idx is not None and prompt_idx < len(prompts):
            st.session_state.input_prompt_template = prompts[prompt_idx].input_template
            st.session_state.text_area_value = st.session_state.input_prompt_template
        else:
            st.session_state.input_prompt_template = None
            st.warning("Selected prompt index is out of bounds. Please select a valid prompt.", icon=":material/error:")
    except Exception as e:
        st.error(f"Failed to update text area: {str(e)}", icon=":material/error:")

def generate_random():
    try:
        prompts = fts.get_prompts()
        prompt_template = st.session_state.input_prompt_template
        prompt_idx = st.session_state.input_prompt_idx

        if prompt_template is not None and prompt_idx is not None and prompt_idx < len(prompts):
            dataset_name = fts.GetDataset(
                GetDatasetRequest(
                    id=prompts[prompt_idx].dataset_id
                )
            ).dataset.huggingface_name

            if dataset_name:
                dataset = load_dataset(dataset_name)
                if "train" in dataset:
                    dataset = dataset["train"]
                idx = random.randint(0, len(dataset) - 1)
                prompt_string = prompt_template.format(**dataset[idx])
                st.session_state.input_prompt = prompt_string
                st.session_state.input_prompt_key = prompt_string
            else:
                st.warning("Dataset not found for the selected prompt.", icon=":material/error:")
        else:
            st.warning("Invalid prompt template or index. Please select a valid prompt.", icon=":material/error:")
    except Exception as e:
        st.error(f"Failed to generate random prompt: {str(e)}", icon=":material/error:")

@st.fragment
def prompt_fragment():
    try:
        cont = st.container()

        prompts = fts.get_prompts()

        if prompts:
            cont.selectbox(
                "Import Prompt Template",
                range(len(prompts)),
                key="input_prompt_idx_key",
                index=st.session_state.input_prompt_idx,
                format_func=lambda x: f"{prompts[x].name} [dataset: {fts.GetDataset(GetDatasetRequest(id=prompts[x].dataset_id)).dataset.name}]",
                on_change=update_text_area
            )
            
            cont.text_area(
                "Prompt Template",
                height=120,
                key="text_area_value",
                value=st.session_state.input_prompt_template,
                on_change=on_text_area_change
            )

            cont.button("Generate Random Prompt from Dataset and Template", on_click=generate_random, use_container_width=True)
            cont.text_area(
                "Input Prompt",
                height=120,
                key="input_prompt_key",
                value=st.session_state.input_prompt,
                on_change=on_input_prompt_change
            )

        else:
            st.warning("No prompts available. Please ensure prompts are loaded correctly.", icon=":material/error:")
    except Exception as e:
        st.error(f"Error in prompt fragment: {str(e)}", icon=":material/error:")

def evaluate_fragment():
    try:
        cont = st.container()

        generate_button = cont.button("Generate", type="primary", use_container_width=True)
        current_model_metadata = st.session_state.current_model_metadata


        if generate_button:

            if current_model_metadata:
                
                if not st.session_state.input_prompt or st.session_state.input_prompt.strip() == "":
                    st.error("Input Prompt is empty or contains only whitespace. Please provide the Input Prompt in the format of the Prompt Template.", icon=":material/error:")
                    return

                with st.spinner("Loading model..."):
                    bnb_config_dict = fts.ListConfigs(
                        ListConfigsRequest(
                            type=ConfigType.BITSANDBYTES_CONFIG,
                            model_id=current_model_metadata.id
                        )
                    ).configs[0].config
                    bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(**json.loads(bnb_config_dict))

                    if st.session_state.prv_model_metadata != st.session_state.current_model_metadata:
                        st.session_state.current_model = AutoModelForCausalLM.from_pretrained(
                            current_model_metadata.huggingface_model_name, quantization_config=bnb_config, return_dict=True
                        )
                        st.session_state.prv_model_metadata = current_model_metadata
                        st.session_state.loaded_adapters = []
                        st.session_state.adapter_outputs = {}

                with st.spinner("Loading Adapters..."):
                    model_adapters = fts.get_adapters()
                    model_adapters = list(filter(lambda x: x.model_id == current_model_metadata.id, model_adapters))
                    model_adapters = list(filter(lambda x: os.path.isdir(x.location), model_adapters))

                    if not model_adapters:
                        st.warning("No adapters found for this model. Please fine-tune the selected base model to generate an adapter.", icon=":material/error:")
                        return

                    st.session_state.model_adapters = model_adapters

                    if hasattr(st.session_state.current_model, 'load_adapter'):
                        for adapter in model_adapters:
                            if adapter.id not in st.session_state.loaded_adapters:
                                loc = adapter.location
                                if os.path.isdir(loc):
                                    st.session_state.current_model.load_adapter(loc, adapter_name=adapter.id)
                                    st.session_state.loaded_adapters.append(adapter.id)
                                else:
                                    st.warning(f"Adapter directory does not exist: {loc}")

                with st.spinner("Generating text..."):
                    tokenizer = AutoTokenizer.from_pretrained(st.session_state.current_model_metadata.huggingface_model_name)
                    input_tokens = tokenizer(st.session_state.input_prompt, return_tensors="pt").to(get_device())

                    st.session_state.current_model.disable_adapters()
                    generation_config_dict = json.loads(st.session_state.generation_config_text or "{}")

                    with torch.amp.autocast('cuda'):
                        model_out = st.session_state.current_model.generate(**input_tokens, **generation_config_dict)

                    tok_out = tokenizer.decode(model_out[0], skip_special_tokens=False)[len(st.session_state.input_prompt):]
                    st.session_state.base_output = tok_out

                    # Generate outputs for each selected adapter
                    st.session_state.adapter_outputs = {}
                    selected_adapters = st.session_state.selected_adapters or []
                    if selected_adapters:
                        for adapter in selected_adapters:
                            st.session_state.current_model.enable_adapters()
                            st.session_state.current_model.set_adapter(adapter.id)
                            with torch.amp.autocast('cuda'):
                                model_out = st.session_state.current_model.generate(**input_tokens, **generation_config_dict)
                            tok_out_adapter = tokenizer.decode(model_out[0], skip_special_tokens=False)[len(st.session_state.input_prompt):]
                            st.session_state.adapter_outputs[adapter.name] = tok_out_adapter
                    else:
                        st.warning("No adapters selected for text generation.", icon=":material/error:")
            else:
                st.error("No model selected. Please select a base model.", icon=":material/error:")
    except Exception as e:
        st.error(f"Error during evaluation: {str(e)}", icon=":material/error:")

# UI Layout
with st.container(border=True):
    col1, col2 = st.columns([1, 17])
    with col1:
        col1.image("./resources/images/difference_24dp_EA3323_FILL0_wght400_GRAD0_opsz40.png")
    with col2:
        col2.subheader('Local Adapter Comparison', divider='red')
        st.caption("Compare your fine-tuned model performance with base model and gain valuable insights.")

st.write("\n")

col1, col2 = st.columns([1, 1])

with col1:
    model_idx = None
    with st.container(border=True):
        try:
            current_models = fts.get_models()
            if current_models:
                model_idx = st.selectbox(
                    "Base Models",
                    range(len(current_models)),
                    format_func=lambda x: current_models[x].name,
                    index=st.session_state.model_idx,
                    key='selected_model_idx',
                    on_change=on_model_change
                )

                if model_idx is not None:
                    st.session_state.current_model_metadata = current_models[model_idx]

                    model_adapters = fts.get_adapters()
                    model_adapters = list(filter(lambda x: x.model_id == current_models[model_idx].id, model_adapters))

                    if model_adapters:
                        selected_adapters = st.multiselect(
                            "Choose Adapters",
                            model_adapters,
                            format_func=lambda x: x.name,
                            key="selected_adapters_key_name",
                            default=st.session_state.selected_adapters,
                            on_change=on_adapters_change
                        )
                        st.session_state.selected_adapters = selected_adapters or []
                    else:
                        st.warning("No adapters available for the selected model. Please fine-tune this model to generate adapters.", icon=":material/error:")
                        st.session_state.selected_adapters = []

                else:
                    st.session_state.base_output = ""
                    st.session_state.adapter_outputs = {}
            else:
                st.error("No models available. Please ensure models are loaded correctly.", icon=":material/error:")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}", icon=":material/error:")

        if model_idx and st.session_state.current_model_metadata and st.session_state.selected_adapters:
            prompt_fragment()
            evaluate_fragment()

with col2:
    if st.session_state.current_model_metadata:
        with st.expander("Generation Arguments:"):
            try:
                generation_config_text = st.text_area(
                    "Generation Config",
                    json.dumps(
                        json.loads(
                            fts.ListConfigs(
                                ListConfigsRequest(
                                    type=ConfigType.GENERATION_CONFIG,
                                    model_id=st.session_state.current_model_metadata.id
                                )
                            ).configs[0].config
                        ),
                        indent=2
                    ),
                    height=200,
                    key='generation_config_text'
                )
            except Exception as e:
                st.error(f"Failed to load generation configuration: {str(e)}", icon=":material/error:")

    cont = st.container(border=True)
    cont.markdown("**Inference Results**")
    if st.session_state.base_output != "":
        cont.text(f"[model: {st.session_state.current_model_metadata.name}]")
        cont.code(st.session_state.base_output)
    else:
        cont.text(f"Base Model Response:")
        cont.text_area("Base Model Response Empty", value="", disabled=True, key="base_output_empty", label_visibility="collapsed", height=200)

    cont.write("\n")

    if st.session_state.selected_adapters:
        for adapter_name, output in st.session_state.adapter_outputs.items():
            cont.markdown("---")
            cont.text(f"[adapter: {adapter_name}]")
            if output:
                cont.code(output)
            else:
                cont.text_area(f"Base+Adapter Response Empty", value="", disabled=True, key=f"{adapter_name}_output_empty", label_visibility="collapsed", height=200)

