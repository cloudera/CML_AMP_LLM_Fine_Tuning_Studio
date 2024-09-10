import streamlit as st
from ft.api import *
import json
from ft.utils import get_env_variable, fetch_resource_usage_data, process_resource_usage_data, get_axolotl_training_config_template_yaml_str, fetch_cml_site_config
from pgs.streamlit_utils import get_fine_tuning_studio_client, get_cml_client
import yaml
from ft.consts import IconPaths, DIVIDER_COLOR

# Instantiate the client to the FTS gRPC app server.
fts = get_fine_tuning_studio_client()
cml = get_cml_client()

cdsw_api_url = get_env_variable('CDSW_API_URL')
cdsw_api_key = get_env_variable('CDSW_API_KEY')
cdsw_project_url = get_env_variable('CDSW_PROJECT_URL')
project_owner = get_env_variable('PROJECT_OWNER', 'User')

# Set up state trackers for this page
if 'ft_adapter_name' not in st.session_state:
    st.session_state['ft_adapter_name'] = ""
if 'ft_output_dir' not in st.session_state:
    st.session_state['ft_output_dir'] = "data/adapters/"
if 'ft_base_model' not in st.session_state:
    st.session_state['ft_base_model'] = None
if 'ft_prompt' not in st.session_state:
    st.session_state['ft_prompt'] = None
if 'ft_dataset' not in st.session_state:
    st.session_state['ft_dataset'] = None
if 'ft_resource_cpu' not in st.session_state:
    st.session_state['ft_resource_cpu'] = 2
if 'ft_resource_mem' not in st.session_state:
    st.session_state['ft_resource_mem'] = 8
if 'ft_resource_gpu' not in st.session_state:
    st.session_state['ft_resource_gpu'] = 1
if 'ft_resource_gpu_label' not in st.session_state:
    st.session_state['ft_resource_gpu_label'] = 1
if 'ft_resource_num_workers' not in st.session_state:
    st.session_state['ft_resource_num_workers'] = 1
if 'ft_dataset_fraction' not in st.session_state:
    st.session_state['ft_dataset_fraction'] = 1.0
if 'ft_dataset_split' not in st.session_state:
    st.session_state['ft_dataset_split'] = 0.9
if 'ft_num_epochs' not in st.session_state:
    st.session_state['ft_num_epochs'] = 10
if 'ft_learning_rate' not in st.session_state:
    st.session_state['ft_learning_rate'] = "2e-4"

# Container for header


def create_header():
    with st.container(border=True):
        col1, col2 = st.columns([1, 17])
        with col1:
            st.image(IconPaths.Experiments.TRAIN_NEW_ADAPTER)
        with col2:
            st.subheader('Train a new Adapter', divider=DIVIDER_COLOR)
            st.caption(
                "Finetune your model using the imported datasets and models, leveraging advanced techniques to improve performance."
            )

# Container for model and adapter selection


def create_train_adapter_page_with_proprietary():
    ccol1, ccol2 = st.columns([4, 2])
    with ccol1:
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                adapter_name = st.text_input(
                    "Adapter Name",
                    placeholder="Adapter name",
                    value=st.session_state['ft_adapter_name'],
                    key="adapter_name",
                    help="Enter the name of the adapter to be created. This field is required."
                )
                st.session_state['ft_adapter_name'] = adapter_name

            with col2:
                adapter_output_dir = st.text_input(
                    "Output Location",
                    value=st.session_state['ft_output_dir'],
                    key="output_location",
                    help="Specify the directory where the adapter will be saved."
                )
                st.session_state['ft_output_dir'] = adapter_output_dir

            # Container for dataset and prompt selection
            col1, col2 = st.columns(2)

            with col1:
                current_datasets = fts.get_datasets()
                dataset_idx = st.selectbox(
                    "Dataset",
                    range(len(current_datasets)),
                    format_func=lambda x: current_datasets[x].name,
                    index=st.session_state['ft_dataset'],
                    key="proprietary_dataset_selectbox",
                    help="Select the dataset to use for training. This field is required."
                )
                st.session_state['ft_dataset'] = dataset_idx
                if dataset_idx is not None:
                    dataset = current_datasets[dataset_idx]
                    current_prompts = fts.get_prompts()
                    current_prompts = list(filter(lambda x: x.dataset_id == dataset.id, current_prompts))
                    prompt_idx = st.selectbox(
                        "Prompts",
                        range(len(current_prompts)),
                        format_func=lambda x: current_prompts[x].name,
                        index=st.session_state['ft_prompt'],
                        key="proprietary_prompt_selectbox",
                        help="Select the prompt to use with the selected dataset. This field is required."
                    )
                    st.session_state['ft_prompt'] = prompt_idx

            with col2:
                current_models = fts.get_models()
                model_idx = st.selectbox(
                    "Base Models",
                    range(len(current_models)),
                    format_func=lambda x: current_models[x].name,
                    index=st.session_state['ft_base_model'],
                    key="proprietary_model_selectbox",
                    help="Select the base model for training. This field is required."
                )
                st.session_state['ft_base_model'] = model_idx
                if dataset_idx is not None:
                    if len(current_prompts) == 0:
                        st.warning(
                            "No prompts available. Please create a prompt template for the selected dataset to proceed with training.",
                            icon=":material/error:")

                    if prompt_idx is not None:
                        st.code(current_prompts[prompt_idx].prompt_template)
            st.divider()
            # Resource Options
            if 'ft_multi_node' not in st.session_state:
                st.session_state['ft_multi_node'] = False
            c1, c2 = st.columns([1, 1])
            with c1:
                ft_type_name = st.selectbox(
                    "Finetuning Type",
                    options=["Single Node", "Multi Node"],
                    index=0,
                    key="ft_type",
                    help="Select whether finetuning runs in standalone mode or distributed across  many processes"
                )
                st.session_state['ft_multi_node'] = ft_type_name == "Multi Node"
            with c2:
                st.empty()
            with st.container(border=True):
                num_workers = st.session_state['ft_resource_num_workers']
                if st.session_state['ft_multi_node']:
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.caption("Distributed Finetuning Units")
                        num_workers = st.number_input(
                            "Number of Distribution Units",
                            min_value=1,
                            max_value=100,
                            value=st.session_state['ft_resource_num_workers'],
                            help="Specify the number of machines that will launched in CML to perform training.")
                        st.session_state['num_workers'] = num_workers
                    with c2:
                        tc_arch = "Distribution - Multi Node"
                        tc_descrip = "**Distributed Finetuning Units** may be provisioned across multiple physical nodes in the CML Workspace."
                        st.info("%s" % (tc_descrip))
                c1, c2 = st.columns([1, 1])
                with c1:
                    resource_label_suffix = ""
                    if st.session_state['ft_multi_node']:
                        resource_label_suffix = " per Distribution Unit"
                    cpu = st.number_input(
                        "CPU" + resource_label_suffix,
                        value=st.session_state['ft_resource_cpu'],
                        min_value=1,
                        key="cpu",
                        help="Specify the number of virtual CPUs to allocate for training."
                    )
                    st.session_state['ft_resource_cpu'] = cpu
                    # Handling the potential for heterogeneous GPU clusters, need to load gpu max num limits and label namess
                    # This is all very heavy, probably want to do this on page load and cache
                    # it once in a more reasonable data struct
                    accelerator_labels = []
                    try:
                        accelerator_labels = cml.list_all_accelerator_node_labels().accelerator_node_label
                        accelerator_labels_dict = {x.label_value: vars(x) for x in accelerator_labels}
                    except Exception as e:
                        site_conf = fetch_cml_site_config(cdsw_api_url, project_owner, cdsw_api_key)
                        site_max_gpu = site_conf.get("max_gpu_per_engine")
                        # Dummy accelerator label that will get ignored in older clusters without
                        # heterogeneous gpu support
                        accelerator_labels_dict = {'Default': {'_availability': True,
                                                               '_id': '-1',
                                                               '_label_value': 'Default',
                                                               '_max_gpu_per_workload': site_max_gpu,
                                                               }}
                        # By default this is 1 to handle the heterogeneous support in CML, but
                        # needs to be 0 for older CML
                        st.session_state['ft_resource_gpu_label'] = 0
                    gpu_label_text_list = [d['_label_value'] for d in accelerator_labels_dict.values()]
                    gpu_label = st.selectbox("GPU Type", options=gpu_label_text_list,
                                             index=st.session_state['ft_resource_gpu_label'])
                    st.session_state['ft_resource_gpu_label'] = gpu_label_text_list.index(gpu_label)
                    gpu_num_max = int(accelerator_labels_dict[gpu_label]['_max_gpu_per_workload'])
                    gpu_label_id = int(accelerator_labels_dict[gpu_label]['_id'])
                with c2:
                    placeholder = st.empty()
                    memory = st.number_input(
                        "Memory (GiB)" + resource_label_suffix,
                        value=st.session_state['ft_resource_mem'],
                        min_value=1,
                        key="memory",
                        help="Specify the amount of memory (in GiB) to allocate for training."
                    )
                    st.session_state['ft_resource_mem'] = memory
                    gpu = st.number_input(
                        "GPU" + resource_label_suffix,
                        min_value=1,
                        value=st.session_state['ft_resource_gpu'],
                        max_value=gpu_num_max,
                        key="gpu",
                        help="Select the number of GPUs to allocate for training. This is limited to the maximum number of GPUs available per node on this CML Workspace Cluster."
                    )
                    st.session_state['ft_resource_gpu'] = gpu

            # Training Options
            with st.container(border=True):
                st.caption("Training Options")
                c1, c2 = st.columns(2)
                with c1:
                    num_epochs = st.number_input(
                        "Number of Epochs",
                        value=st.session_state['ft_num_epochs'],
                        min_value=1,
                        key="num_epochs",
                        help="Specify the number of epochs for training."
                    )
                    st.session_state['ft_num_epochs'] = num_epochs
                with c2:
                    learning_rate = st.text_input(
                        "Learning Rate",
                        value=st.session_state['ft_learning_rate'],
                        key="learning_rate",
                        help="Set the learning rate for the training process."
                    )
                    st.session_state['ft_learning_rate'] = learning_rate
                c1, c2 = st.columns([1, 1])
                dataset_fraction = c1.slider(
                    "Dataset Fraction",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state['ft_dataset_fraction'],
                    help="Specify the fraction of the dataset to use for training. If this is set to anything other"
                    " than 1.0, the 'train' dataset split will be downsampled with random sampling by this factor."
                    " note that this does not affect the size of the 'test' or 'eval' split if those splits are"
                    " in a dataset."
                )
                st.session_state['ft_dataset_fraction'] = dataset_fraction
                dataset_train_test_split = c2.slider(
                    "Dataset Train/Test Split",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state['ft_dataset_split'],
                    help="Set the ratio for splitting the dataset into training and test splits. NOTE: this setting"
                    " only applies to datasets that ONLY have a 'train' split. If there is a 'test' or an 'eval' split, then"
                    " that split will be used if the training run requires evaluation at epoch boundaries. If there"
                    " is only a 'train' split in the dataset, then the dataset will be split between a training"
                    " and a test dataset, AFTER the dataset has been optionally sampled by the dataset fraction.")
                st.session_state['ft_dataset_split'] = dataset_train_test_split

                c1, c2 = st.columns([1, 1])

                # Extract out the lora config and the bnb config to use. For now,
                # server-side there is no selection logic based on model & adapters,
                # but there may be in the future, which is why we are specifying this here.
                # Right now, we are just extracting out the first available config and
                # showing that to the user.
                with c1:
                    with st.expander("LoRA Config"):
                        if 'ft_config_lora' not in st.session_state:
                            st.session_state['ft_config_lora'] = json.dumps(
                                json.loads(
                                    fts.ListConfigs(
                                        ListConfigsRequest(
                                            type=ConfigType.LORA_CONFIG,
                                            model_id=current_models[model_idx].id) if model_idx else ListConfigsRequest(
                                            type=ConfigType.LORA_CONFIG)).configs[0].config),
                                indent=2)

                        lora_config_text = st.text_area(
                            "",
                            value=st.session_state['ft_config_lora'],
                            height=200,
                            help="LoRA configuration for fine-tuning the model.")
                        st.session_state['ft_config_lora'] = lora_config_text
                with c2:
                    with st.expander("BitsAndBytes Config"):
                        if 'ft_config_bnb' not in st.session_state:
                            st.session_state['ft_config_bnb'] = json.dumps(
                                json.loads(
                                    fts.ListConfigs(
                                        ListConfigsRequest(
                                            type=ConfigType.BITSANDBYTES_CONFIG,
                                            model_id=current_models[model_idx].id) if model_idx else ListConfigsRequest(
                                            type=ConfigType.BITSANDBYTES_CONFIG)).configs[0].config),
                                indent=2)

                        bnb_config_text = st.text_area(
                            "",
                            value=st.session_state['ft_config_bnb'],
                            height=200,
                            help="BitsAndBytes configuration for optimizing model training.")

                with st.expander("Advanced Training Options"):
                    st.info("""
                            NOTE: the following fields in the below JSON will be overridden by the values set in the UI above:
                            * Output Location (TrainingArguments.output_dir)
                            * Number of Epochs (TrainingArguments.num_train_epochs)
                            * Learning Rate (TrainingArguments.learning_rate)
                            """)
                    if 'ft_config_trainer' not in st.session_state:
                        st.session_state['ft_config_trainer'] = json.dumps(
                            json.loads(
                                fts.ListConfigs(
                                    ListConfigsRequest(
                                        type=ConfigType.TRAINING_ARGUMENTS,
                                        model_id=current_models[model_idx].id) if model_idx else ListConfigsRequest(
                                        type=ConfigType.TRAINING_ARGUMENTS)).configs[0].config),
                            indent=2)
                    training_args_text = st.text_area(
                        "Training Arguments",
                        value=st.session_state['ft_config_trainer'],
                        height=400,
                        help="Advanced training arguments in JSON format.")

            # Start job button
            button_enabled = dataset_idx is not None and model_idx is not None and prompt_idx is not None and adapter_name != ""
            start_job_button = st.button(
                "Start Job",
                type="primary",
                use_container_width=True,
                key="start_job_button",
                help="Start the fine-tuning job. Ensure all required fields are filled in before starting."
            )

            if start_job_button:
                if not button_enabled:
                    st.warning(
                        "Please complete the fields: Adapter Name, Dataset, Prompt, and Base Model before starting the job.",
                        icon=":material/error:")
                else:
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
                                type=ConfigType.LORA_CONFIG,
                                config=lora_config_text,
                                description=model.huggingface_model_name
                            )
                        ).config
                        bnb_config: ConfigMetadata = fts.AddConfig(
                            AddConfigRequest(
                                type=ConfigType.BITSANDBYTES_CONFIG,
                                config=bnb_config_text,
                                description=model.huggingface_model_name
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
                                type=ConfigType.TRAINING_ARGUMENTS,
                                config=json.dumps(training_args_config_dict),
                                description=model.huggingface_model_name
                            )
                        ).config

                        fts.StartFineTuningJob(
                            StartFineTuningJobRequest(
                                adapter_name=adapter_name,
                                base_model_id=model.id,
                                dataset_id=dataset.id,
                                prompt_id=prompt.id,
                                num_workers=int(num_workers),
                                auto_add_adapter=True,
                                num_epochs=int(num_epochs),
                                learning_rate=float(learning_rate),
                                cpu=int(cpu),
                                gpu=gpu,
                                gpu_label_id=int(gpu_label_id),
                                memory=int(memory),
                                model_bnb_config_id=bnb_config.id,
                                adapter_bnb_config_id=bnb_config.id,
                                lora_config_id=lora_config.id,
                                training_arguments_config_id=training_args_config.id,
                                output_dir=adapter_output_dir,
                                dataset_fraction=dataset_fraction,
                                train_test_split=dataset_train_test_split
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

    with ccol2:
        st.info("""
        ### How to Train a Model ?

        1. **Fill in the unique adapter name, base model, dataset, and prompts**: These are essential fields required to initiate the training process. Ensure each of these fields is correctly filled.

        2. **Output Location**: Specify the directory where the trained adapter will be saved. This directory will be created or will exist within the AMP project's directory.

        3. **Compute Configuration (CPU, Memory, GPU)**: Indicate the number of virtual CPUs (vCPUs), memory (in GiB), and GPUs to allocate for the training job in the CML Workspace. Ensure the configuration is sufficient to handle the model and dataset size efficiently, providing adequate resources to prevent errors and accelerate the training process.

        4. **LoRA Config**: This configuration is used for Low-Rank Adaptation (LoRA), a technique to fine-tune models with fewer parameters, making the process more efficient. Review and adjust this configuration as necessary to match the training needs of your model.

        5. **BitsAndBytes Config**: The BitsAndBytes configuration helps optimize model training by enabling efficient computation and memory usage. Adjust these settings to enhance the performance of the model during training, particularly in resource-constrained environments.

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

# TODO : This will be displayed in UI, after end to end support for inference and MLflow evaluation.


def create_train_adapter_page_with_axolotl():
    ccol1, ccol2 = st.columns([3, 2])
    with ccol1:
        with st.container(border=True):

            col1, col2 = st.columns(2)
            with col1:
                adapter_name = st.text_input(
                    "Adapter Name",
                    placeholder="Adapter name",
                    key="adapter_name_axolotl",
                    help="Enter the name of the adapter to be created. This field is required."
                )

            with col2:
                adapter_output_dir = st.text_input(
                    "Output Location",
                    value="data/adapters/",
                    key="output_location_axolotl",
                    help="Specify the directory where the adapter will be saved."
                )

            current_models = fts.get_models()
            model_idx = st.selectbox(
                "Base Models",
                range(len(current_models)),
                format_func=lambda x: current_models[x].name,
                index=None,
                key="axolotl_model_selectbox",
                help="Select the base model for training. This field is required."
            )

            # Container for dataset and prompt selection
            col1, col2 = st.columns(2)
            current_datasets = fts.get_datasets()
            dataset_idx = col1.selectbox(
                "Dataset",
                range(len(current_datasets)),
                format_func=lambda x: current_datasets[x].name,
                index=None,
                key="axolotl_dataset_selectbox",
                help="Select the dataset to use for training. This field is required."
            )

            dataset_format_idx = None  # Initialize dataset_format_idx to avoid NameError
            max_matching_keys = 0  # To track the maximum number of matching keys

            if dataset_idx is not None:
                dataset = current_datasets[dataset_idx]
                dataset_features = json.loads(dataset.features) if dataset.features else []
                col1.code("\n * " + '\n * '.join(dataset_features))

                current_dataset_formats = fts.ListConfigs(
                    ListConfigsRequest(type=ConfigType.AXOLOTL_DATASET_FORMATS)
                ).configs

                for idx, dataset_format in enumerate(current_dataset_formats):
                    format_config = json.loads(dataset_format.config)
                    matching_keys = len(set(format_config.keys()) & set(dataset_features))

                    if matching_keys > max_matching_keys:
                        max_matching_keys = matching_keys
                        dataset_format_idx = idx

                if dataset_format_idx is not None and max_matching_keys > 0:
                    st.info(
                        "**Note:** A suitable dataset format has been auto-selected for training. Please verify or choose the correct type from the Axolotl-supported options. [**More info**](https://axolotl-ai-cloud.github.io/axolotl/docs/dataset-formats/inst_tune.html).")

                    dataset_format_idx = col2.selectbox(
                        "Dataset Types",
                        range(len(current_dataset_formats)),
                        format_func=lambda x: current_dataset_formats[x].description,
                        index=dataset_format_idx,
                        key="axolotl_dataset_format_selectbox",
                        help="Select the format of the dataset. This field is required."
                    )

                    dataset_format = current_dataset_formats[dataset_format_idx]
                    format_config = json.loads(dataset_format.config)
                    col2.code(json.dumps(format_config, indent=2), "json")
                else:
                    st.error(
                        "This is an unsupported dataset type. Please use a proprietary solution for training.",
                        icon=":material/error:")

            # Advanced options
            c1, c2 = st.columns(2)
            with c1:
                num_epochs = st.text_input(
                    "Number of Epochs",
                    value="10",
                    key="num_epochs_axolotl",
                    help="Specify the number of epochs for training."
                )
            with c2:
                learning_rate = st.text_input(
                    "Learning Rate",
                    value="2e-4",
                    key="learning_rate_axolotl",
                    help="Set the learning rate for the training process."
                )

            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                cpu = st.text_input(
                    "CPU (vCPU)",
                    value="2",
                    key="cpu_axolotl",
                    help="Specify the number of virtual CPUs to allocate for training."
                )
            with c2:
                memory = st.text_input(
                    "Memory (GiB)",
                    value="8",
                    key="memory_axolotl",
                    help="Specify the amount of memory (in GiB) to allocate for training."
                )
            with c3:
                gpu = st.selectbox(
                    "GPU (NVIDIA)",
                    options=[1],
                    index=0,
                    key="gpu_axolotl",
                    help="Select the number of GPUs to allocate for training."
                )

            auto_add_adapter = True
            # st.checkbox(
            #    "Add Adapter to Fine Tuning Studio after Training",
            #    value=True,
            #    key="auto_add_adapter_axolotl",
            #    help="Automatically add the trained adapter to the Fine Tuning Studio after the job completes."
            # )

            val_set_size = st.slider(
                "Validation Set Size",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                help="Specify the fraction of the dataset to be used as the validation set."
            )

            st.info("""
                - Go to [**Axolotl Training Examples**](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples).
                - Choose an example model, such as LLaMA-3, and then select a specific configuration file, like `qlora.yml`.
                - Copy the contents of the YAML file and paste it directly into the Axolotl Training Arguments field in the form.
                - Once you've pasted the configuration, simply click the "Start Job" button to begin the training process.
                - **Note:** In the Axolotl Training Arguments below, the YAML fields **base_model**, **datasets.path**, **datasets.type**, **num_epochs**, **learning_rate** and **val_set_size** will be overridden by the values specified above.
            """)

            axolotl_training_args_text = st.text_area(
                "**Axolotl Training Arguments**",
                get_axolotl_training_config_template_yaml_str(),
                height=400,
                help="Advanced training arguments in YAML format."
            )

            button_enabled = dataset_idx is not None and dataset_format_idx is not None and model_idx is not None and adapter_name != ""
            start_job_button = st.button(
                "Start Job",
                type="primary",
                use_container_width=True,
                key="start_job_button_axolotl",
                help="Start the fine-tuning job. Ensure all required fields are filled in before starting."
            )

            if start_job_button:
                if not button_enabled:
                    st.warning(
                        "Please complete the fields: Adapter Name, Dataset, Dataset Type, and Base Model before starting the job.",
                        icon=":material/error:")
                else:
                    try:
                        model = current_models[model_idx]
                        dataset = current_datasets[dataset_idx]

                        training_args_config_dict = None
                        # Parse the text input as YAML
                        try:
                            # First, try to parse as YAML
                            training_args_config_dict = yaml.safe_load(axolotl_training_args_text)
                            training_args_config_dict['num_epochs'] = num_epochs
                            training_args_config_dict['learning_rate'] = learning_rate
                            training_args_config_dict['val_set_size'] = val_set_size
                            # Ensure 'datasets' key exists and is a list
                            if 'datasets' not in training_args_config_dict:
                                training_args_config_dict['datasets'] = []

                            # Ensure the first item in the 'datasets' list exists and is a dictionary
                            if len(training_args_config_dict['datasets']) == 0:
                                training_args_config_dict['datasets'].append({})

                            # Ensure the first item is a dictionary (if it isn't already)
                            if not isinstance(training_args_config_dict['datasets'][0], dict):
                                training_args_config_dict['datasets'][0] = {}

                            # Now safely set the 'type' key
                            training_args_config_dict['datasets'][0]['type'] = current_dataset_formats[dataset_format_idx].description

                        except Exception as e:
                            st.error(
                                f"Axolotl Training Arguments is not a valid YAML: **{str(e)}**",
                                icon=":material/error:")
                            st.toast(
                                f"Axolotl Training Arguments is not a valid YAML: **{str(e)}**",
                                icon=":material/error:")

                            # Convert the dictionary back to YAML for storing in the configuration
                        axolotl_train_config_yaml = yaml.dump(training_args_config_dict, default_flow_style=False)

                        if training_args_config_dict is not None:
                            # Add the configuration
                            axolotl_train_config = fts.AddConfig(
                                AddConfigRequest(
                                    type=ConfigType.AXOLOTL,
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
                                    axolotl_config_id=axolotl_train_config.id,
                                    framework_type=FineTuningFrameworkType.AXOLOTL
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
                            st.error(f"Axolotl Training Arguments cannot be empty.", icon=":material/error:")
                            st.toast(f"Axolotl Training Arguments cannot be empty.", icon=":material/error:")
                    except Exception as e:
                        st.error(f"Failed to create Finetuning Job: **{str(e)}**", icon=":material/error:")
                        st.toast(f"Failed to Finetuning Job: **{str(e)}**", icon=":material/error:")

    with ccol2:
        st.info("""
        ### How to Train a Model with Axolotl

        1. **Fill in the unique adapter name, base model, dataset, and dataset type**: These are essential fields required to initiate the training process. Ensure each of these fields is correctly filled.

        2. **Dataset Format and Features**: The selected dataset type/format must match the dataset's features. Refer to the [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/docs/dataset-formats/inst_tune.html) to ensure that the dataset format is compatible with the features of the dataset you have chosen.

        3. **Compute Configuration (CPU, Memory, GPU)**: Indicate the number of virtual CPUs (vCPUs), memory (in GiB), and GPUs to allocate for the training job in the CML Workspace. Ensure the configuration is sufficient to handle the model and dataset size efficiently, providing adequate resources to prevent errors and accelerate the training process.

        4. **Axolotl Training Arguments**: To fill in the Axolotl Training Arguments:

            - Below are the Axolotl Training Arguments, which are explained in more detail at the following link:
            [**Axolotl Configuration Documentation**](https://github.com/axolotl-ai-cloud/axolotl/blob/main/docs/config.qmd).
            - You can also check out various examples of Axolotl Training Arguments for multiple models and fine-tuning techniques here:
            [**Axolotl Training Examples**](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples).

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

create_train_adapter_page_with_proprietary()
