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

# Instantiate (or get the pre-existing) client to the FTS gRPC app server.
fts = get_fine_tuning_studio_client()


# Initialize session state. This is the recommended
# way to initialize session state parameters in Streamlit.
# https://docs.streamlit.io/develop/concepts/architecture/session-state
if 'current_model_metadata' not in st.session_state:
    st.session_state.current_model_metadata = None
if 'model_adapter' not in st.session_state:
    st.session_state.model_adapter = None
if 'base_output' not in st.session_state:
    st.session_state.base_output = ""
if 'base_output2' not in st.session_state:
    st.session_state.base_output2 = ""
if 'current_model' not in st.session_state:
    st.session_state.current_model = None


def update_text_area():
    prompts = fts.get_prompts()
    prompt_idx = st.session_state.input_prompt_idx

    if prompt_idx is not None:
        st.session_state.input_prompt_template = prompts[prompt_idx].prompt_template
    else:
        st.session_state.input_prompt_template = None


def generate_random():
    prompts = fts.get_prompts()
    prompt_template = st.session_state.input_prompt_template
    prompt_idx = st.session_state.input_prompt_idx

    if prompt_template is not None and prompt_idx is not None:
        dataset = load_dataset(fts.GetDataset(
            GetDatasetRequest(
                id=prompts[prompt_idx].dataset_id
            )
        ).dataset.huggingface_name)
        if "train" in dataset:
            dataset = dataset["train"]
        idx = random.randint(0, len(dataset) - 1)
        prompt_string = prompt_template.format(**dataset[idx])
        st.session_state.input_prompt = prompt_string


@st.fragment
def prompt_fragment():
    cont = st.container()

    prompts = fts.get_prompts()

    prompt_idx = cont.selectbox(
        "Import Prompt Template",
        range(len(prompts)),
        key="input_prompt_idx",
        format_func=lambda x: f"{prompts[x].name} [dataset: {fts.GetDataset(GetDatasetRequest(id=prompts[x].dataset_id)).dataset.name}]",
        index=None,
        on_change=update_text_area)
    cont.text_area("Prompt Template", height=120, key="input_prompt_template")

    gen_random = cont.button("Generate Random Prompt from Dataset and Template",
                             on_click=generate_random, use_container_width=True)
    cont.text_area("Input Prompt", height=120, key="input_prompt")


def evaluate_fragment():
    cont = st.container()

    generate_button = cont.button("Generate", type="primary", use_container_width=True)

    if generate_button:
        with st.spinner("Generating text..."):
            tokenizer = AutoTokenizer.from_pretrained(st.session_state.current_model_metadata.huggingface_model_name)

            input_tokens = tokenizer(st.session_state.input_prompt, return_tensors="pt").to(get_device())

            # Disable the model adapters so we can see baseline mdoel performance.
            st.session_state.current_model.disable_adapters()

            generation_config_dict = json.loads(st.session_state.generation_config_text)

            with torch.amp.autocast('cuda'):
                model_out = st.session_state.current_model.generate(
                    **input_tokens,
                    **generation_config_dict,
                )

            tok_out = tokenizer.decode(model_out[0], skip_special_tokens=False)[len(st.session_state.input_prompt):]
            st.session_state.base_output = tok_out

            # Enable adapters and set them equal to the adapter names, which
            # we have set as the adapter id.
            st.session_state.current_model.enable_adapters()
            st.session_state.current_model.set_adapter(st.session_state.model_adapter.id)
            with torch.amp.autocast('cuda'):
                model_out = st.session_state.current_model.generate(
                    **input_tokens,
                    **generation_config_dict,
                )
            tok_out2 = tokenizer.decode(model_out[0], skip_special_tokens=False)[len(st.session_state.input_prompt):]
            st.session_state.base_output2 = tok_out2


# Attempt to log in to huggingface for gated models
hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
attempt_hf_login(hf_token)

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
    with st.container(border=True):
        current_models = fts.get_models()
        model_idx = st.selectbox(
            "Base Models",
            range(len(current_models)),
            format_func=lambda x: current_models[x].name,
            index=None
        )

        if model_idx is not None:
            current_model_metadata = current_models[model_idx]

            if st.session_state.current_model_metadata != current_model_metadata:
                with st.spinner("Loading model..."):

                    # Grab a BnB config for this model. Note that the ListConfigs call
                    # does not currently filter configs based on model/adapter id's, but
                    # may do so in the future. For now, just take the first available
                    # BnB config from the list.
                    bnb_config_dict = json.loads(
                        fts.ListConfigs(
                            ListConfigsRequest(
                                type=ConfigType.BITSANDBYTES_CONFIG,
                                model_id=current_model_metadata.id
                            )
                        ).configs[0].config
                    )
                    bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(**bnb_config_dict)

                    st.session_state.current_model = AutoModelForCausalLM.from_pretrained(
                        current_model_metadata.huggingface_model_name, quantization_config=bnb_config, return_dict=True)
                    st.session_state.current_model_metadata = current_model_metadata

            model_adapters: List[AdapterMetadata] = fts.get_adapters()
            model_adapters = list(filter(lambda x: x.model_id == current_model_metadata.id, model_adapters))

            # Filter adapters based on their presence in the /data/adapter directory
            model_adapters = list(filter(lambda x: os.path.isdir(os.path.join(x.location)), model_adapters))

            if 'adapters_loaded' not in st.session_state or not st.session_state.adapters_loaded:
                with st.spinner("Loading Adapters..."):
                    for adapter in model_adapters:
                        loc = os.path.join(adapter.location)
                        if not loc.endswith("/"):
                            loc += "/"

                        if os.path.isdir(loc):
                            if hasattr(st.session_state.current_model, 'load_adapter'):
                                st.session_state.current_model.load_adapter(loc, adapter_name=adapter.id)
                            else:
                                raise ValueError("Not supported!")
                    st.session_state.adapters_loaded = True

            model_adapter_idx = st.selectbox(
                "Choose an Adapter",
                range(len(model_adapters)),
                format_func=lambda x: model_adapters[x].name,
                index=None
            )

            if model_adapter_idx is not None:
                model_adapter = model_adapters[model_adapter_idx]
                st.session_state.model_adapter = model_adapter
                st.session_state.model_idx = model_idx
                st.session_state.model_adapter_idx = model_adapter_idx

        if st.session_state.current_model_metadata and st.session_state.model_adapter:
            prompt_fragment()
            evaluate_fragment()


with col2:

    # Generation config expander
    if st.session_state.current_model_metadata and st.session_state.model_adapter:
        with st.expander("Generation Arguments:"):

            # Extract out the generation args for this model. Right now, there is no config-selection
            # logic based on model id or otherwise, but there may be in the future. For now, just take
            # the first generation config in the config store and use it here.
            generation_config_text = st.text_area(
                "Generation Config",
                json.dumps(
                    json.loads(
                        fts.ListConfigs(
                            ListConfigsRequest(
                                type=ConfigType.GENERATION_CONFIG,
                                model_id=st.session_state.current_model_metadata.id,
                                adapter_id=st.session_state.model_adapter.id
                            )
                        ).configs[0].config
                    ),
                    indent=2
                ),
                height=200,
                key='generation_config_text'
            )

    cont = st.container(border=True)
    if st.session_state.current_model_metadata:
        cont.text(f"Base [model: {st.session_state.current_model_metadata.name}]")
    else:
        cont.text(f"Base Model Response")
    if st.session_state.base_output != "":
        cont.code(st.session_state.base_output)
    else:
        cont.text_area(f"Base Model Response Empty", value="", disabled=True,
                       key="base_output_empty", label_visibility="collapsed", height=200)

    cont.write("\n")

    if st.session_state.model_adapter:
        cont.text(f"Base+Adapter [adapter: {st.session_state.model_adapter.name}]")
    else:
        cont.text(f"Base+Adapter Response")

    if st.session_state.base_output2 != "":
        cont.code(st.session_state.base_output2)
    else:
        cont.text_area(f"Base+Adapter Response Empty", value="", disabled=True,
                       key="base_output2_empty", label_visibility="collapsed", height=200)
