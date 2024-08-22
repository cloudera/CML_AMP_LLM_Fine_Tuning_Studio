import streamlit as st
from ft.api import *
import json
from ft.utils import get_env_variable, fetch_resource_usage_data, process_resource_usage_data
import traceback
from pgs.streamlit_utils import get_fine_tuning_studio_client
import yaml

# Custom representer to handle empty strings
def represent_none(self, _):
    return self.represent_scalar('tag:yaml.org,2002:null', '')

# Register the custom representer for NoneType
yaml.add_representer(type(None), represent_none)

# Instantiate the client to the FTS gRPC app server.
fts = get_fine_tuning_studio_client()


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


def create_train_adapter_page_with_proprietary():
    ccol1, ccol2 = st.columns([3, 2])
    with ccol1:
        with st.container(border=True):

            col1, col2 = st.columns(2)
            with col1:
                adapter_name = st.text_input("Adapter Name", placeholder="Adapter name", key="adapter_name")

            with col2:
                adapter_output_dir = st.text_input("Output Location", value="data/adapters/", key="output_location")

            # Container for dataset and prompt selection
            col1, col2 = st.columns(2)

            with col1:
                current_datasets = fts.get_datasets()
                dataset_idx = st.selectbox(
                    "Dataset",
                    range(len(current_datasets)),
                    format_func=lambda x: current_datasets[x].name,
                    index=None,
                    key="proprietary_dataset_selectbox"
                )
                if dataset_idx is not None:
                    dataset = current_datasets[dataset_idx]
                    current_prompts = fts.get_prompts()
                    current_prompts = list(filter(lambda x: x.dataset_id == dataset.id, current_prompts))
                    prompt_idx = st.selectbox(
                        "Prompts",
                        range(len(current_prompts)),
                        format_func=lambda x: current_prompts[x].name,
                        index=None,
                        key="proprietary_prompt_selectbox"
                    )

                    if prompt_idx is not None:
                        st.code(current_prompts[prompt_idx].prompt_template)

            with col2:
                current_models = fts.get_models()
                model_idx = st.selectbox(
                    "Base Models",
                    range(len(current_models)),
                    format_func=lambda x: current_models[x].name,
                    index=None,
                    key="proprietary_model_selectbox"
                )

            # Advanced options
            c1, c2 = st.columns(2)
            with c1:
                num_epochs = st.text_input("Number of Epochs", value="10", key="num_epochs")
            with c2:
                learning_rate = st.text_input("Learning Rate", value="2e-4", key="learning_rate")
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                cpu = st.text_input("CPU (vCPU)", value="2", key="cpu")
            with c2:
                memory = st.text_input("Memory (GiB)", value="8", key="memory")
            with c3:
                gpu = st.selectbox("GPU (NVIDIA)", options=[1], index=0, key="gpu")

            auto_add_adapter = st.checkbox(
                "Add Adapter to Fine Tuning Studio after Training",
                value=True,
                key="auto_add_adapter")
            c1, c2 = st.columns([1, 1])
            dataset_fraction = c1.slider("Dataset Fraction", min_value=0.0, max_value=1.0, value=1.0)
            dataset_train_test_split = c2.slider("Dataset Train/Test Split", min_value=0.0, max_value=1.0, value=0.9)

            c1, c2 = st.columns([1, 1])

            # Extract out the lora config and the bnb config to use. For now,
            # server-side there is no selection logic based on model & adapters,
            # but there may be in the future, which is why we are specifying this here.
            # Right now, we are just extracting out the first available config and
            # showing that to the user.
            lora_config_text = c1.text_area(
                "LoRA Config",
                json.dumps(
                    json.loads(
                        fts.ListConfigs(
                            ListConfigsRequest(type=ConfigType.CONFIG_TYPE_LORA_CONFIG)
                        ).configs[0].config
                    ),
                    indent=2
                ),
                height=200
            )
            bnb_config_text = c2.text_area(
                "BitsAndBytes Config",
                json.dumps(
                    json.loads(
                        fts.ListConfigs(
                            ListConfigsRequest(type=ConfigType.CONFIG_TYPE_BITSANDBYTES_CONFIG)
                        ).configs[0].config
                    ),
                    indent=2
                ),
                height=200
            )

            with st.expander("Advanced Training Options"):
                st.info("""
                        NOTE: the following fields in the below JSON will be overridden by the values set in the UI above:
                        * Output Location (TrainingArguments.output_dir)
                        * Number of Epochs (TrainingArguments.num_train_epochs)
                        * Learning Rate (TrainingArguments.learning_rate)
                        """)
                training_args_text = st.text_area(
                    "Training Arguments",
                    json.dumps(
                        json.loads(
                            fts.ListConfigs(
                                ListConfigsRequest(type=ConfigType.CONFIG_TYPE_TRAINING_ARGUMENTS)
                            ).configs[0].config
                        ),
                        indent=2
                    ),
                    height=400
                )

            # Start job button
            button_enabled = dataset_idx is not None and model_idx is not None and prompt_idx is not None and adapter_name != ""
            start_job_button = st.button(
                "Start Job",
                type="primary",
                use_container_width=True,
                disabled=not button_enabled,
                key="start_job_button"
            )

            if start_job_button:
                try:
                    # Extract out relevant model metadata.
                    model = current_models[model_idx]
                    dataset = current_datasets[dataset_idx]
                    prompt = current_prompts[prompt_idx]

                    # If we've made changes to our configs, let's update them with FTS so other
                    # components of FTS can access the configs. Note: if a pre-existing config
                    # exactly matches these configs, the metadata (id) of the pre-existing config
                    # will be returned.
                    lora_config: ConfigMetadata = fts.AddConfig(
                        AddConfigRequest(
                            type=ConfigType.CONFIG_TYPE_LORA_CONFIG,
                            config=lora_config_text
                        )
                    ).config
                    bnb_config: ConfigMetadata = fts.AddConfig(
                        AddConfigRequest(
                            type=ConfigType.CONFIG_TYPE_BITSANDBYTES_CONFIG,
                            config=bnb_config_text
                        )
                    ).config

                    # Override the fields that were set in the UI. Note that the
                    # output dir passed to the training job should only become
                    # available when we have a UUID, which becomes available
                    # further down the road. Because of this, we are adding multiple
                    # training args configs to the config store right off the bat, which
                    # we may want to clean later.
                    training_args_config_dict = json.loads(training_args_text)
                    training_args_config_dict["learning_rate"] = float(learning_rate)
                    training_args_config_dict["num_train_epochs"] = int(num_epochs)

                    training_args_config: ConfigMetadata = fts.AddConfig(
                        AddConfigRequest(
                            type=ConfigType.CONFIG_TYPE_TRAINING_ARGUMENTS,
                            config=json.dumps(training_args_config_dict)
                        )
                    ).config

                    fts.StartFineTuningJob(
                        StartFineTuningJobRequest(
                            adapter_name=adapter_name,
                            base_model_id=model.id,
                            dataset_id=dataset.id,
                            prompt_id=prompt.id,
                            num_workers=int(1),
                            auto_add_adapter=auto_add_adapter,
                            num_epochs=int(num_epochs),
                            learning_rate=float(learning_rate),
                            cpu=int(cpu),
                            gpu=gpu,
                            memory=int(memory),
                            model_bnb_config_id=bnb_config.id,
                            adapter_bnb_config_id=bnb_config.id,
                            lora_config_id=lora_config.id,
                            training_arguments_config_id=training_args_config.id,
                            output_dir=adapter_output_dir,
                            dataset_fraction=dataset_fraction,
                            train_test_split=dataset_train_test_split,
                            finetuning_framework_type=FinetuningFrameworkType.FINETUNING_FRAMEWORK_TYPE_PROPRIETARY_SOLUTION
                        )
                    )
                    st.success(
                        "Create Finetuning Job. Please go to **Monitor Training Job** tab!",
                        icon=":material/check:"
                    )
                    st.toast(
                        "Create Finetuning Job. Please go to **Monitor Training Job** tab!",
                        icon=":material/check:"
                    )
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


def create_train_adapter_page_with_axolotl():
    ccol1, ccol2 = st.columns([3, 2])
    with ccol1:
        with st.container(border=True):

            col1, col2 = st.columns(2)
            with col1:
                adapter_name = st.text_input("Adapter Name", placeholder="Adapter name", key="adapter_name_axolotl")

            with col2:
                adapter_output_dir = st.text_input(
                    "Output Location",
                    value="data/adapters/",
                    key="output_location_axolotl")

            # Container for dataset and prompt selection
            col1, col2 = st.columns(2)

            with col1:
                current_datasets = fts.get_datasets()
                dataset_idx = st.selectbox(
                    "Dataset",
                    range(len(current_datasets)),
                    format_func=lambda x: current_datasets[x].name,
                    index=None,
                    key="axolotl_dataset_selectbox"
                )

            with col2:
                current_models = fts.get_models()
                model_idx = st.selectbox(
                    "Base Models",
                    range(len(current_models)),
                    format_func=lambda x: current_models[x].name,
                    index=None,
                    key="axolotl_model_selectbox"
                )

            # Advanced options
            c1, c2 = st.columns(2)
            with c1:
                num_epochs = st.text_input("Number of Epochs", value="10", key="num_epochs_axolotl")
            with c2:
                learning_rate = st.text_input("Learning Rate", value="2e-4", key="learning_rate_axolotl")

            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                cpu = st.text_input("CPU (vCPU)", value="2", key="cpu_axolotl")
            with c2:
                memory = st.text_input("Memory (GiB)", value="8", key="memory_axolotl")
            with c3:
                gpu = st.selectbox("GPU (NVIDIA)", options=[1], index=0, key="gpu_axolotl")

            auto_add_adapter = st.checkbox(
                "Add Adapter to Fine Tuning Studio after Training",
                value=True,
                key="auto_add_adapter_axolotl")

            st.info("""
                - Below are the Axolotl Training Arguments, which are explained in more detail at the following link: 
                [**Axolotl Configuration Documentation**](https://github.com/axolotl-ai-cloud/axolotl/blob/main/docs/config.qmd).
                - You can also check out various examples of Axolotl Training Arguments for multiple models and fine-tuning techniques here: 
                [**Axolotl Training Examples**](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples).
                - **Note:** In the Axolotl Training Arguments below, the YAML fields **base_model**, **datasets.path**, **num_epochs**, and **learning_rate** will be overridden by the values specified above.
            """)

            axolotl_training_args_text = st.text_area(
                "**Axolotl Training Arguments**",
                yaml.dump(
                    yaml.safe_load(
                        fts.ListConfigs(
                            ListConfigsRequest(type=ConfigType.CONFIG_TYPE_AXOLOTL_TRAIN_CONFIG)
                        ).configs[0].config
                    ),
                    indent=2,
                    default_flow_style=False
                ),
                height=400
            )

            button_enabled = dataset_idx is not None and model_idx is not None and adapter_name != ""
            start_job_button = st.button(
                "Start Job",
                type="primary",
                use_container_width=True,
                disabled=not button_enabled,
                key="start_job_button_axolotl"
            )

            if start_job_button:
                try:
                    model = current_models[model_idx]
                    dataset = current_datasets[dataset_idx]

                    training_args_config_dict = None
                    # Parse the text input as YAML
                    try:
                        # First, try to parse as YAML
                        training_args_config_dict = yaml.safe_load(axolotl_training_args_text)
                    except Exception as e:
                        st.error(f"Axolotl Training Arguments is not a valid YAML: **{str(e)}**", icon=":material/error:")
                        st.toast(f"Axolotl Training Arguments is not a valid YAML: **{str(e)}**", icon=":material/error:")

                        # Convert the dictionary back to YAML for storing in the configuration
                    axolotl_train_config_yaml = yaml.dump(training_args_config_dict, default_flow_style=False)

                    if training_args_config_dict is not None:
                        # Add the configuration
                        axolotl_train_config = fts.AddConfig(
                            AddConfigRequest(
                                type=ConfigType.CONFIG_TYPE_AXOLOTL_TRAIN_CONFIG,
                                config=axolotl_train_config_yaml
                            )
                        ).config

                        fts.StartFineTuningJob(
                            StartFineTuningJobRequest(
                                adapter_name=adapter_name,
                                base_model_id=model.id,
                                dataset_id=dataset.id,
                                num_workers=int(1),
                                auto_add_adapter=auto_add_adapter,
                                num_epochs=int(num_epochs),
                                learning_rate=float(learning_rate),
                                cpu=int(cpu),
                                gpu=gpu,
                                memory=int(memory),
                                output_dir=adapter_output_dir,
                                axolotl_train_config_id=axolotl_train_config.id,
                                finetuning_framework_type=FinetuningFrameworkType.FINETUNING_FRAMEWORK_TYPE_AXOLOTL
                            )
                        )
                        st.success(
                            "Create Finetuning Job. Please go to **Monitor Training Job** tab!",
                            icon=":material/check:"
                        )
                        st.toast(
                            "Create Finetuning Job. Please go to **Monitor Training Job** tab!",
                            icon=":material/check:"
                        )
                    else:
                        st.error(f"Axolotl Training Arguments can not be empty.", icon=":material/error:")
                        st.toast(f"Axolotl Training Arguments can not be empty.", icon=":material/error:")
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
                use_container_width=True,
                key="axolotl_resource_usage"
            )


create_header()
tab1, tab2 = st.tabs(["**Train with Proprietary Solution**", "**Train with Axolotl**"])
with tab1:
    create_train_adapter_page_with_proprietary()
with tab2:
    create_train_adapter_page_with_axolotl()
