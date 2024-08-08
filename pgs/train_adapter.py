import traceback
import streamlit as st
from ft.app import get_app
from ft.state import get_state
from ft.api import *
import json
from ft.utils import get_env_variable, fetch_resource_usage_data, process_resource_usage_data

cdsw_api_url = get_env_variable('CDSW_API_URL')
cdsw_api_key = get_env_variable('CDSW_API_KEY')
cdsw_project_url = get_env_variable('CDSW_PROJECT_URL')
project_owner = get_env_variable('PROJECT_OWNER', 'User')


# Container for header
def create_header():
    with st.container(border=True):
        col1, col2 = st.columns([1, 17])
        with col1:
            st.image("./resources/images/forward_24dp_EA3323_FILL0_wght400_GRAD0_opsz40.png")
        with col2:
            st.subheader('Train a new Adapter', divider='red')
            st.caption(
                "Finetune your model using the imported datasets and models, leveraging advanced techniques to improve performance.")

# Container for model and adapter selection


def create_train_adapter_page():
    ccol1, ccol2 = st.columns([3, 2])
    with ccol1:
        with st.container(border=True):

            col1, col2 = st.columns(2)
            with col1:
                adapter_name = st.text_input("Adapter Name", placeholder="Adapter name", key="adapter_name")

            with col2:
                current_models = get_state().models
                model_idx = st.selectbox(
                    "Base Models",
                    range(
                        len(current_models)),
                    format_func=lambda x: current_models[x].name,
                    index=None)

            # Container for dataset and prompt selection
            col1, col2 = st.columns(2)

            with col1:
                current_datasets = get_state().datasets
                dataset_idx = st.selectbox(
                    "Dataset",
                    range(
                        len(current_datasets)),
                    format_func=lambda x: current_datasets[x].name,
                    index=None)
                if dataset_idx is not None:
                    dataset = current_datasets[dataset_idx]
                    current_prompts = get_state().prompts
                    current_prompts = list(filter(lambda x: x.dataset_id == dataset.id, current_prompts))
                    prompt_idx = st.selectbox(
                        "Prompts",
                        range(
                            len(current_prompts)),
                        format_func=lambda x: current_prompts[x].name,
                        index=None)

                    if prompt_idx is not None:
                        st.code(current_prompts[prompt_idx].prompt_template)

            with col2:
                adapter_location = st.text_input("Output Location", value="data/adapters/", key="output_location")

            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                cpu = st.text_input("CPU (vCPU)", value="2", key="cpu")
            with c2:
                memory = st.text_input("Memory (GiB)", value="8", key="memory")
            with c3:
                gpu = st.selectbox("GPU (NVIDIA)", options=[1], index=0)

            # Advanced options
            c1, c2 = st.columns(2)
            with c1:
                num_epochs = st.text_input("Number of Epochs", value="10", key="num_epochs")
            with c2:
                learning_rate = st.text_input("Learning Rate", value="2e-4", key="learning_rate")

            framework = st.selectbox("Select Fine-Tuning Framework", ["Legacy", "Axolotl"])
            st.markdown("---")
            if framework == "Legacy":
                c1, c2 = st.columns([1, 1])
                lora_config = c1.text_area(
                    "LoRA Config",
                    json.dumps(
                        json.load(
                            open(".app/configs/default_lora_config.json")),
                        indent=2),
                    height=200)
                bnb_config = c2.text_area(
                    "BitsAndBytes Config",
                    json.dumps(
                        json.load(
                            open(".app/configs/default_bnb_config.json")),
                        indent=2),
                    height=200)

                # Start job button
                button_enabled = dataset_idx is not None and model_idx is not None and prompt_idx is not None and adapter_name != ""
                start_job_button = st.button(
                    "Start Job",
                    type="primary",
                    use_container_width=True,
                    disabled=not button_enabled)

                if start_job_button:
                    try:
                        model = current_models[model_idx]
                        dataset = current_datasets[dataset_idx]
                        prompt = current_prompts[prompt_idx]
                        bnb_config_dict = json.loads(bnb_config)
                        bnb_config_special_type: BnbConfig = BnbConfig(**bnb_config_dict)
                        get_app().launch_ft_job(StartFineTuningJobRequest(
                            adapter_name=adapter_name,
                            base_model_id=model.id,
                            dataset_id=dataset.id,
                            prompt_id=prompt.id,
                            num_workers=int(1),
                            bits_and_bytes_config=bnb_config_special_type,
                            auto_add_adapter=True,
                            num_epochs=int(num_epochs),
                            learning_rate=float(learning_rate),
                            cpu=int(cpu),
                            gpu=gpu,
                            memory=int(memory),
                            finetuning_framework='legacy'
                        ))
                        st.success(
                            "Create Finetuning Job. Please go to **Monitor Training Job** tab!",
                            icon=":material/check:")
                        st.toast(
                            "Create Finetuning Job. Please go to **Monitor Training Job** tab!",
                            icon=":material/check:")
                    except Exception as e:
                        st.error(f"Failed to create Finetuning Job: **{str(e)}**", icon=":material/error:")
                        st.toast(f"Failed to Finetuning Job: **{str(e)}**", icon=":material/error:")
                        print(traceback.format_exc())
                        st.error(traceback.format_exc())
            elif framework == "Axolotl":
                col1, col2, _, col3 = st.columns([5, 5, 1, 5])
                model_type = col1.selectbox(
                    "Model Type",
                    [
                        "AutoModelForCausalLM",
                        "LlamaForCausalLM",
                        "MambaLMHeadModel",
                        "GPTNeoXForCausalLM",
                        "MistralForCausalLM"
                    ]
                )

                # Text input fields
                tokenizer_type = col2.selectbox(
                    "Tokenizer Type",
                    [
                        "AutoTokenizer",
                        "LlamaTokenizer",
                        "CodeLlamaTokenizer",
                        "GPT2Tokenizer"
                    ]
                )

                adapter_type = col1.selectbox(
                    "Adapter Type",
                    [
                        "lora",
                        "qlora"
                    ]
                )

                dataset_type = col2.selectbox(
                    "Dataset Type",
                    [
                        "alpaca",
                        "jeopardy",
                        "chat_template",
                        "completion",
                        "alpaca:chat",
                        "alpaca:phi",
                        "chat_template.argilla"
                    ]
                )

                # Selectbox for quantization options
                quantization = col3.selectbox("Quantization", ["None", "8-bit", "4-bit"])

                # Set values based on quantization selection
                load_in_8bit = quantization == "8-bit"
                load_in_4bit = quantization == "4-bit"

                val_set_size = col1.number_input("Validation Set Size", min_value=0.0, max_value=1.0, value=0.05)

                # Sequence length fields
                sequence_len = col2.number_input("Sequence Length", min_value=1, value=4096)

                # Adapter settings
                lora_r = col3.number_input("LoRA r", min_value=1, value=32)
                lora_alpha = col3.number_input("LoRA Alpha", min_value=1, value=16)
                lora_dropout = col3.number_input("LoRA Dropout", min_value=0.0, max_value=1.0, value=0.05)
                lora_target_modules = col3.text_input("LoRA Target Modules", "")
                lora_target_linear = col3.checkbox("LoRA Target Linear", True)

                # Selectbox for scheduler and optimizer
                lr_scheduler = col1.selectbox("Learning Rate Scheduler", ["cosine", "one_cycle", "log_sweep"], index=0)
                optimizer = col2.selectbox(
                    "Optimizer",
                    [
                        "adamw_hf",
                        "adamw_torch",
                        "adamw_torch_fused",
                        "adamw_torch_xla",
                        "adamw_apex_fused",
                        "adafactor",
                        "adamw_anyprecision",
                        "sgd",
                        "adagrad",
                        "adamw_bnb_8bit",
                        "lion_8bit",
                        "lion_32bit",
                        "paged_adamw_32bit",
                        "paged_adamw_8bit",
                        "paged_lion_32bit",
                        "paged_lion_8bit",
                        "galore_adamw",
                        "galore_adamw_8bit",
                        "galore_adafactor",
                        "galore_adamw_layerwise",
                        "galore_adamw_8bit_layerwise",
                        "galore_adafactor_layerwise"
                    ]
                )

                attention = col1.selectbox("Attention", ["Flash Attention", "X-Former Attention", "None"])

                # Set values based on quantization selection
                flash_attention = attention == "Flash Attention"
                xformers_attention = attention == "X-Former Attention"

                # Start job button
                button_enabled = dataset_idx is not None and model_idx is not None and prompt_idx is not None and adapter_name != ""
                start_job_button = st.button(
                    "Start Job",
                    type="primary",
                    use_container_width=True,
                    disabled=not button_enabled)

                if start_job_button:
                    try:
                        model = current_models[model_idx]
                        dataset = current_datasets[dataset_idx]
                        prompt = current_prompts[prompt_idx]
                        dataset_config = DatasetConfig(path="", type=dataset_type, train_on_split="true")
                        get_app().launch_ft_job(StartFineTuningJobRequest(
                            adapter_name=adapter_name,
                            base_model_id=model.id,
                            dataset_id=dataset.id,
                            prompt_id=prompt.id,
                            num_workers=int(1),
                            auto_add_adapter=True,
                            num_epochs=int(num_epochs),
                            learning_rate=float(learning_rate),
                            cpu=int(cpu),
                            gpu=gpu,
                            memory=int(memory),
                            finetuning_framework='axolotl',
                            axolotl_train_config=AxolotlTrainConfig(
                                base_model="",  # str
                                model_type=model_type,  # str
                                tokenizer_type=tokenizer_type,  # str
                                adapter=adapter_type,  # str
                                datasets=[dataset_config],  # List[DatasetConfig]
                                load_in_8bit=bool(load_in_8bit),  # bool
                                load_in_4bit=bool(load_in_4bit),  # bool
                                val_set_size=float(val_set_size),  # float
                                sequence_len=int(sequence_len),  # int
                                lora_r=int(lora_r),  # int
                                lora_alpha=int(lora_alpha),  # int
                                lora_dropout=float(lora_dropout),  # float
                                lora_target_modules=lora_target_modules,  # ListOfString
                                lora_target_linear=bool(lora_target_linear),  # bool
                                lr_scheduler=lr_scheduler,  # str
                                optimizer=optimizer,  # str
                                num_epochs=int(num_epochs),  # int
                                learning_rate=float(learning_rate),  # float
                                trust_remote_code=False,  # bool
                                tokenizer_use_fast=True,  # bool
                                tokenizer_legacy=True,  # bool
                                gptq=False,  # bool
                                bf16="auto",  # str
                                fp16=False,  # bool
                                tf32=False,  # bool
                                shuffle_merged_datasets=True,  # bool
                                dataset_prepared_path="axolotl/last_run_prepared",  # str
                                pad_to_sequence_len=True,  # bool
                                sample_packing=True,  # bool
                                eval_sample_packing=False,  # bool
                                lora_model_dir="",  # str
                                mlflow_tracking_uri="",  # str
                                mlflow_experiment_name="",  # str
                                hf_mlflow_log_artifacts=False,  # bool
                                output_dir="",  # str
                                gradient_accumulation_steps=int(4),  # int
                                micro_batch_size=int(2),  # int
                                warmup_ratio=float(0.05),  # float
                                logging_steps=int(1),  # int
                                evals_per_epoch=int(4),  # int
                                train_on_inputs=False,  # bool
                                group_by_length=False,  # bool
                                gradient_checkpointing=True,  # bool
                                weight_decay=float(0.0),  # float
                                strict=False,  # bool
                                flash_attention=bool(flash_attention),  # bool
                                xformers_attention=bool(xformers_attention)  # bool
                            )

                        ))
                        st.success(
                            "Create Finetuning Job. Please go to **Monitor Training Job** tab!",
                            icon=":material/check:")
                        st.toast(
                            "Create Finetuning Job. Please go to **Monitor Training Job** tab!",
                            icon=":material/check:")
                    except Exception as e:
                        st.error(f"Failed to create Finetuning Job: **{str(e)}**", icon=":material/error:")
                        st.toast(f"Failed to Finetuning Job: **{str(e)}**", icon=":material/error:")
                        print(traceback.format_exc())
                        st.error(traceback.format_exc())

    with ccol2:
        st.info("""
        ### Advanced Options

        - **CPU**: Enter the number of CPU cores to be used for training.
        - **Memory**: Specify the amount of memory (in GiB) to be allocated for the training process.
        - **GPU**: Select the number of GPUs to be used. Currently, only one GPU option is available.
        - **LoRA Config**: Provide the configuration for LoRA (Low-Rank Adaptation), which is a technique used to fine-tune models with fewer parameters.
        - **BitsAndBytes Config**: Provide the configuration for the BitsAndBytes library, which can optimize model performance during training.

        ### Starting the Job

        The button will be enabled only when all required fields are filled out.
        """)
        ccol2.caption("**Resource Usage**")
        data = fetch_resource_usage_data(cdsw_api_url, project_owner, cdsw_api_key)
        if data:
            df = process_resource_usage_data(data)
            st.data_editor(
                df[['Resource Name', 'Progress', 'Max Available']],
                column_config={
                    "Resource Name": "Resource Name",
                    "Progress": st.column_config.ProgressColumn(
                        "Usage",
                        help="Current Resource Consumption",
                        format="%.0f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "Max Available": "Available (Cluster)"
                },
                hide_index=True,
                use_container_width=True
            )


create_header()
create_train_adapter_page()
