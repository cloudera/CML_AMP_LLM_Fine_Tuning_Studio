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
from ft.consts import IconPaths, DIVIDER_COLOR

# Instantiate (or get the pre-existing) client to the FTS gRPC app server.
fts = get_fine_tuning_studio_client()

# Initialize session state with default values
default_session_state = {
    'current_model_metadata': None,
    'prv_model_metadata': None,
    'model_adapters': [],
    'base_output': "",
    'adapter_outputs': {},
    'current_model': None,
    'loaded_adapters': [],
    'generation_config_text': "",
    'input_prompt': "",
    'input_prompt_template': None,
    'completion_template': None,
    'completion_string': "",
    'input_prompt_idx': None,
    'selected_adapters': [],
    'model_idx': None,
    'locked_prompt_id': None
}

for key, default_value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# Handle Huggingface login attempt
hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
if hf_token:
    attempt_hf_login(hf_token)


def on_model_change():
    st.session_state.model_idx = st.session_state.selected_model_idx
    st.session_state.selected_adapters = []  # Reset selected adapters when model changes
    st.session_state.locked_prompt_id = None  # Reset the locked prompt ID when the model changes

    # Reload all adapters for the newly selected model
    if st.session_state.current_model_metadata:
        st.session_state.model_adapters = [
            adapter for adapter in fts.get_adapters() if adapter.model_id == st.session_state.current_model_metadata.id
        ]


def on_adapters_change():
    selected_adapters = st.session_state.selected_adapters_key_name
    st.session_state.selected_adapters = selected_adapters

    if selected_adapters:
        # Lock the prompt ID to the first selected adapter's prompt_id
        if st.session_state.locked_prompt_id is None:
            st.session_state.locked_prompt_id = selected_adapters[0].prompt_id

        # Filter the available adapters to only those with the same prompt_id
        st.session_state.filtered_adapters = [
            adapter for adapter in st.session_state.model_adapters
            if adapter.prompt_id == st.session_state.locked_prompt_id
        ]

        # Filter prompts based on the locked prompt_id
        st.session_state.filtered_prompts = [
            prompt for prompt in fts.get_prompts() if prompt.id == st.session_state.locked_prompt_id
        ]

        if st.session_state.filtered_prompts:
            st.session_state.input_prompt_idx = 0
            st.session_state.input_prompt_template = st.session_state.filtered_prompts[0].input_template
            st.session_state.completion_template = st.session_state.filtered_prompts[0].completion_template
            st.session_state.text_area_value = st.session_state.input_prompt_template
        else:
            st.session_state.filtered_prompts = []
            st.session_state.input_prompt_template = None
            st.session_state.completion_template = None
            st.session_state.text_area_value = None

    else:
        # Reset filtered adapters to all model adapters when no adapters are selected
        st.session_state.filtered_adapters = st.session_state.model_adapters
        st.session_state.locked_prompt_id = None


def on_text_area_change():
    st.session_state.input_prompt_template = st.session_state.text_area_value


def on_input_prompt_change():
    st.session_state.input_prompt = st.session_state.input_prompt_key


def update_text_area():
    """Update the text area with the selected prompt template and generate the completion string."""
    if st.session_state.input_prompt_idx is not None:
        prompt_idx = st.session_state.input_prompt_idx
        if prompt_idx < len(st.session_state.filtered_prompts):
            st.session_state.input_prompt_template = st.session_state.filtered_prompts[prompt_idx].input_template
            st.session_state.completion_template = st.session_state.filtered_prompts[prompt_idx].completion_template
            st.session_state.text_area_value = st.session_state.input_prompt_template

            # Generate the completion string based on the prompt and completion templates
            try:
                dataset_name = fts.GetDataset(
                    GetDatasetRequest(
                        id=st.session_state.filtered_prompts[prompt_idx].dataset_id
                    )
                ).dataset.huggingface_name

                if dataset_name:
                    dataset = load_dataset(dataset_name)
                    if "train" in dataset:
                        dataset = dataset["train"]
                    idx = random.randint(0, len(dataset) - 1)

                    prompt_string = st.session_state.input_prompt_template.format(**dataset[idx])
                    completion_string = st.session_state.completion_template.format(**dataset[idx])

                    st.session_state.input_prompt = prompt_string
                    st.session_state.completion_string = completion_string
                else:
                    st.warning("Dataset not found for the selected prompt.", icon=":material/error:")
            except Exception as e:
                st.error(f"Failed to generate completion string: {str(e)}", icon=":material/error:")

        else:
            st.warning("Selected prompt index is out of bounds. Please select a valid prompt.", icon=":material/error:")


def generate_random():
    try:
        prompt_template = st.session_state.input_prompt_template
        completion_template = st.session_state.completion_template
        if prompt_template and completion_template:
            dataset_name = fts.GetDataset(
                GetDatasetRequest(
                    id=st.session_state.filtered_prompts[0].dataset_id
                )
            ).dataset.huggingface_name

            if dataset_name:
                dataset = load_dataset(dataset_name)
                if "train" in dataset:
                    dataset = dataset["train"]
                idx = random.randint(0, len(dataset) - 1)
                prompt_string = prompt_template.format(**dataset[idx])
                completion_string = completion_template.format(**dataset[idx])
                st.session_state.input_prompt = prompt_string
                st.session_state.input_prompt_key = prompt_string
                st.session_state.completion_string = completion_string
            else:
                st.warning("Dataset not found for the selected prompt.", icon=":material/error:")
        else:
            st.warning("Invalid prompt or completion template. Please select a valid prompt.", icon=":material/error:")
    except Exception as e:
        st.error(f"Failed to generate random prompt: {str(e)}", icon=":material/error:")


@st.fragment
def prompt_fragment():
    try:
        cont = st.container(border=True)

        if st.session_state.filtered_prompts:
            # Updated heading for text_area to include Prompt name and Dataset name
            prompt_name = st.session_state.filtered_prompts[st.session_state.input_prompt_idx].name
            dataset_name = fts.GetDataset(
                GetDatasetRequest(id=st.session_state.filtered_prompts[st.session_state.input_prompt_idx].dataset_id)
            ).dataset.name

            cont.markdown(f"**Prompt**  :  {prompt_name} [Dataset : {dataset_name}]")
            cont.text_area(
                f"Prompt Template",
                height=120,
                key="text_area_value",
                value=st.session_state.input_prompt_template,
                on_change=on_text_area_change
            )

            cont.button("Generate Random Prompt from Dataset and Template",
                        on_click=generate_random, use_container_width=True)

            cont.text_area(
                "Input Prompt",
                height=120,
                key="input_prompt_key",
                value=st.session_state.input_prompt,
                on_change=on_input_prompt_change
            )

            # Display the generated completion string
            if st.session_state.completion_string:
                cont.caption("Expected Output")
                cont.code(st.session_state.completion_string, language="text")

        else:
            st.warning("No prompts available. Please select an adapter with a valid prompt.", icon=":material/error:")
    except Exception as e:
        st.error(f"Error in prompt fragment: {str(e)}", icon=":material/error:")


def evaluate_fragment():
    try:
        cont = st.container()

        if st.session_state.current_model_metadata:
            with cont.expander("Generation Config"):
                try:
                    generation_config_text = st.text_area(
                        "",
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
                        height=220,
                        key='generation_config_text'
                    )
                except Exception as e:
                    st.error(f"Failed to load generation configuration: {str(e)}", icon=":material/error:")

        generate_button = cont.button("Generate", type="primary", use_container_width=True)
        current_model_metadata = st.session_state.current_model_metadata

        if generate_button:
            if current_model_metadata:
                if not st.session_state.input_prompt or st.session_state.input_prompt.strip() == "":
                    st.error(
                        "Input Prompt is empty or contains only whitespace. Please provide the Input Prompt in the format of the Prompt Template.",
                        icon=":material/error:")
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
                            current_model_metadata.huggingface_model_name, quantization_config=bnb_config, return_dict=True)
                        st.session_state.prv_model_metadata = current_model_metadata
                        st.session_state.loaded_adapters = []
                        st.session_state.adapter_outputs = {}

                with st.spinner("Loading Adapters..."):
                    model_adapters = fts.get_adapters()
                    model_adapters = list(filter(lambda x: x.model_id == current_model_metadata.id, model_adapters))
                    model_adapters = list(filter(lambda x: os.path.isdir(x.location), model_adapters))

                    if not model_adapters:
                        st.warning(
                            "No adapters found for this model. Please fine-tune the selected base model to generate an adapter.",
                            icon=":material/error:")
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
                    tokenizer = AutoTokenizer.from_pretrained(
                        st.session_state.current_model_metadata.huggingface_model_name)
                    input_tokens = tokenizer(st.session_state.input_prompt, return_tensors="pt").to(get_device())

                    st.session_state.current_model.disable_adapters()
                    generation_config_dict = json.loads(st.session_state.generation_config_text or "{}")

                    with torch.amp.autocast('cuda'):
                        model_out = st.session_state.current_model.generate(**input_tokens, **generation_config_dict)

                    tok_out = tokenizer.decode(model_out[0], skip_special_tokens=False)[
                        len(st.session_state.input_prompt):]
                    st.session_state.base_output = tok_out

                    # Generate outputs for each selected adapter
                    st.session_state.adapter_outputs = {}
                    selected_adapters = st.session_state.selected_adapters or []
                    if selected_adapters:
                        for adapter in selected_adapters:
                            st.session_state.current_model.enable_adapters()
                            st.session_state.current_model.set_adapter(adapter.id)
                            with torch.amp.autocast('cuda'):
                                model_out = st.session_state.current_model.generate(
                                    **input_tokens, **generation_config_dict)
                            tok_out_adapter = tokenizer.decode(model_out[0], skip_special_tokens=False)[
                                len(st.session_state.input_prompt):]
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
        col1.image(IconPaths.Experiments.LOCAL_ADAPTER_COMPARISON)
    with col2:
        col2.subheader('Local Adapter Comparison', divider=DIVIDER_COLOR)
        st.caption("Compare your fine-tuned model performance with base model and gain valuable insights.")

st.write("\n")

col1, _, col2 = st.columns([30, 1, 30])

with col1:
    st.subheader("**Configure Models & Prompts**", divider=DIVIDER_COLOR)
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

                    st.session_state.model_adapters = model_adapters

                    if model_adapters:
                        # Apply initial filtering based on locked prompt_id
                        filtered_adapters = model_adapters
                        if st.session_state.locked_prompt_id:
                            filtered_adapters = [
                                adapter for adapter in model_adapters
                                if adapter.prompt_id == st.session_state.locked_prompt_id
                            ]

                        selected_adapters = st.multiselect(
                            "Choose Adapters",
                            filtered_adapters,
                            format_func=lambda x: x.name,
                            key="selected_adapters_key_name",
                            default=st.session_state.selected_adapters,
                            on_change=on_adapters_change
                        )

                        # Update session state to ensure consistency
                        st.session_state.selected_adapters = selected_adapters or []
                    else:
                        st.warning(
                            "No adapters available for the selected model. Please fine-tune this model to generate adapters.",
                            icon=":material/error:")
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
    st.subheader("**Inference Results**", divider=DIVIDER_COLOR)
    cont = st.container(border=True)
    if st.session_state.base_output != "":
        cont.markdown(f"**Base Model: {st.session_state.current_model_metadata.name}**")
        cont.code(st.session_state.base_output)
    else:
        cont.markdown(f"**Base Model Response:**")
        cont.text_area(
            "Base Model Response Empty",
            value="",
            disabled=True,
            key="base_output_empty",
            label_visibility="collapsed",
            height=200)

    cont.write("\n")

    if st.session_state.selected_adapters:
        for adapter_name, output in st.session_state.adapter_outputs.items():
            cont = st.container(border=True)
            cont.markdown(f"**Adapter: {adapter_name}**")
            if output:
                cont.code(output)
            else:
                cont.text_area(
                    f"Base+Adapter Response Empty",
                    value="",
                    disabled=True,
                    key=f"{adapter_name}_output_empty",
                    label_visibility="collapsed",
                    height=200)
