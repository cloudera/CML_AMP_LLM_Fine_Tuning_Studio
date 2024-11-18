import streamlit as st
import os
from ft.utils import get_env_variable, fetch_resource_usage_data, process_resource_usage_data, fetch_cml_site_config
from typing import List
from ft.api import *
from pgs.streamlit_utils import get_fine_tuning_studio_client, get_cml_client
import json
from ft.consts import IconPaths, DIVIDER_COLOR, BASE_MODEL_ONLY_IDX, BASE_MODEL_ONLY_ADAPTER_ID

# Instantiate the client to the FTS gRPC app server.
fts = get_fine_tuning_studio_client()
cml = get_cml_client()


project_owner = get_env_variable('PROJECT_OWNER', 'User')
cdsw_api_url = get_env_variable('CDSW_API_URL')
cdsw_api_key = get_env_variable('CDSW_API_KEY')
cdsw_project_url = get_env_variable('CDSW_PROJECT_URL')

if 'ft_resource_gpu_label' not in st.session_state:
    st.session_state['ft_resource_gpu_label'] = 1

if 'selected_features' not in st.session_state:
    st.session_state['selected_features'] = []

if 'eval_dataset_fraction' not in st.session_state:
    st.session_state['eval_dataset_fraction'] = 1.0

if 'mlflow_dataset_idx' not in st.session_state:
    st.session_state['mlflow_dataset_idx'] = None

if 'mlflow_model_idx' not in st.session_state:
    st.session_state['mlflow_model_idx'] = None

if 'mlflow_prompt_idx' not in st.session_state:
    st.session_state['mlflow_prompt_idx'] = None

if 'gpu_label' not in st.session_state:
    st.session_state['gpu_label'] = None

if 'mlflow_cpu' not in st.session_state:
    st.session_state['mlflow_cpu'] = "2"

if 'mlflow_memory' not in st.session_state:
    st.session_state['mlflow_memory'] = "8"

# Container for header
with st.container(border=True):
    col1, col2 = st.columns([1, 17])
    with col1:
        col1.image(IconPaths.Experiments.RUN_MLFLOW_EVALUATION)
    with col2:
        col2.subheader('Run MLFlow Evaluation', divider=DIVIDER_COLOR)
        st.caption("Execute comprehensive MLFlow evaluations on your fine-tuned model to ensure accuracy, performance, and reliability, gaining valuable insights.")

st.write("\n")

# Container for model and adapter selection
ccol1, ccol2 = st.columns([3, 2])
with ccol1:
    with st.container(border=True):

        # Container for dataset and prompt selection
        col1, col2 = st.columns(2)
        current_datasets = fts.get_datasets()
        mlflow_dataset_idx = st.selectbox(
            "Datasets",
            range(len(current_datasets)),
            format_func=lambda x: current_datasets[x].name,
            index=st.session_state['mlflow_dataset_idx']
        )

        if mlflow_dataset_idx is not None:
            st.session_state['mlflow_dataset_idx'] = mlflow_dataset_idx
            dataset = current_datasets[mlflow_dataset_idx]
            current_prompts = fts.get_prompts()
            current_prompts = list(filter(lambda x: x.dataset_id == dataset.id, current_prompts))
            mlflow_prompt_idx = st.selectbox(
                "Prompts",
                range(len(current_prompts)),
                format_func=lambda x: current_prompts[x].name,
                index=st.session_state['mlflow_prompt_idx'],
                key="prompt_selectbox",
                help="Select the prompt to use with the selected dataset. This field is required."
            )

            if len(current_prompts) == 0:
                st.error(
                    "No prompts available. Please create a prompt template for the selected dataset to proceed with training.",
                    icon=":material/error:")

            if mlflow_prompt_idx is not None:
                st.session_state['mlflow_prompt_idx'] = mlflow_prompt_idx
                subcol1, subcol2 = st.columns(2)
                subcol1.caption("Prompt Template")
                subcol1.code(current_prompts[mlflow_prompt_idx].input_template)
                subcol2.caption("Completion Template")
                subcol2.code(current_prompts[mlflow_prompt_idx].completion_template)

            eval_dataset_fraction = st.slider(
                "Evaluation Dataset Fraction",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                value=st.session_state['eval_dataset_fraction'],
                help="Specify the fraction of the dataset to use for evaluation. If you want to do a quick evaluation"
                " select a smaller fraction such as 0.1 which will take only 10 percent of the evaluation dataset."
                " For full dataset evaluation, keep it to 1. "
                " Please note, dataset here refers to evaluation dataset. The training rows will not be used for evaluation")
            st.session_state['eval_dataset_fraction'] = eval_dataset_fraction

            dataset_features = json.loads(fts.GetDataset(
                GetDatasetRequest(
                    id=dataset.id
                )
            ).dataset.features)
            selected_features = st.multiselect(
                "Choose Extra Columns That You Need In The Evaluation CSV",
                dataset_features,
                help="These extra columns will be included in the evaluation CSV along with the model input, expected output and the model output columns. Leave it blank for default behaviour.",
                key="selected_dataset_features",
                default=st.session_state['selected_features'],
            )

            st.session_state['selected_features'] = selected_features or []

        st.divider()
        st.caption("**Choose Models for Evaluation**")
        all_model_adapter_combinations = []
        CURRENT_MODEL = None
        NUM_GPUS = 5  # let's make it dynamic via API call to cdp
        # currently let's try to take at max 3 model + adapter combination and dispatch them to cml jobs
        with st.container(border=True):
            for i in range(NUM_GPUS):
                with st.container(border=True):
                    current_models = fts.get_models()
                    mlflow_model_idx = st.selectbox(
                        "Base Model",
                        range(len(current_models)),
                        format_func=lambda x: current_models[x].name,
                        index=st.session_state['mlflow_model_idx'],
                        key=f"current_model_index_{i}"
                    )

                    model_adapter_idx = None
                    model_adapter = None
                    # TODO: this currently assumes HF model for local eval, but should not have to be in the future
                    if mlflow_model_idx is not None:
                        st.session_state['mlflow_model_idx'] = mlflow_model_idx
                        current_model_metadata = current_models[mlflow_model_idx]

                        model_adapters: List[AdapterMetadata] = fts.get_adapters()
                        model_adapters = list(filter(lambda x: x.model_id == current_model_metadata.id, model_adapters))

                        # Filter adapters based on their presence in the /data/adapter directory
                        model_adapters = list(filter(lambda x: os.path.isdir(os.path.join(x.location)), model_adapters))

                        # TODO: We should not have to load the adapters every run, this is overkill
                        with st.spinner("Loading Adapters..."):
                            for adapter in model_adapters:
                                loc = os.path.join(adapter.location)
                                if not loc.endswith("/"):
                                    loc += "/"

                        only = st.toggle("Base Model Only", key=f"base_model_evaluation_only_key_{i}")
                        if not only:
                            model_adapter_idx = st.selectbox(
                                "Choose an Adapter",
                                range(len(model_adapters)),
                                format_func=lambda x: model_adapters[x].name,
                                index=None,
                                key=f"current_adapter_index_{i}"
                            )

                            if len(model_adapters) == 0:
                                st.error(
                                    "No adapters available. Please create a fine tuning job for the selected base model to create an adapter. Or run evaluation on base model only!",
                                    icon=":material/error:")

                            if model_adapter_idx is not None:
                                model_adapter = model_adapters[model_adapter_idx]
                        else:
                            model_adapter = BASE_MODEL_ONLY_IDX
                        st.session_state['model_adapter'] = model_adapter
                        if {"mlflow_model_idx": mlflow_model_idx,
                                "model_adapter": model_adapter} in all_model_adapter_combinations:
                            st.error(
                                "This Model Adapter combination is already selected. This will lead to duplicate evaluation results.")
                if mlflow_model_idx is not None and model_adapter is not None:
                    all_model_adapter_combinations.append(
                        {"mlflow_model_idx": mlflow_model_idx, "model_adapter": model_adapter})
                if i == NUM_GPUS - 1:
                    continue
                add = st.toggle(label="Add", key=f"add_additional_model_{i}")
                if add:
                    continue
                else:
                    break
        # Advanced options
        st.caption("**Advanced Options**")
        c1, c2 = st.columns([1, 1])
        with c1:
            mlflow_cpu = st.text_input("CPU(vCPU)", value=st.session_state['mlflow_cpu'], key="mlflow_cpu")
        with c2:
            mlflow_memory = st.text_input("Memory(GiB)", value=st.session_state['mlflow_memory'], key="mlflow_memory")

        gpu = st.selectbox("GPU(NVIDIA)", options=[1], index=0)
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
            # By default this is 1 to handle the heterogeneous support in CML, but needs to be 0 for older CML
            st.session_state['ft_resource_gpu_label'] = 0
        gpu_label_text_list = [d['_label_value'] for d in accelerator_labels_dict.values()]
        gpu_label = st.selectbox("GPU Type", options=gpu_label_text_list,
                                 index=st.session_state['ft_resource_gpu_label'])
        st.session_state['ft_resource_gpu_label'] = gpu_label_text_list.index(gpu_label)
        gpu_label_id = int(accelerator_labels_dict[gpu_label]['_id'])

        button_enabled = mlflow_dataset_idx is not None and mlflow_model_idx is not None and model_adapter is not None and mlflow_prompt_idx is not None

        if button_enabled:
            with st.expander("Configs"):
                cc1, cc2 = st.columns([1, 1])
                mlflow_model_idx = all_model_adapter_combinations[0]["mlflow_model_idx"]
                model_adapter = all_model_adapter_combinations[0]["model_adapter"]
                if model_adapter is not BASE_MODEL_ONLY_IDX:
                    adapter_id = model_adapter.id
                else:
                    adapter_id = BASE_MODEL_ONLY_ADAPTER_ID
                # Extract out a BnB config and a generation config that will be used for
                # this specific mlflow evaluation run. Right now there is no selection logic on
                # these configs for a specific model type, but there may be in the future. For now,
                # just use the first selected configuration for each.
                # Just picking up the first models configuration.
                bnb_config_text = cc1.text_area(
                    "Quantization Config",
                    json.dumps(
                        json.loads(
                            fts.ListConfigs(
                                ListConfigsRequest(
                                    type=ConfigType.BITSANDBYTES_CONFIG,
                                    model_id=current_models[mlflow_model_idx].id,
                                    adapter_id=adapter_id
                                )
                            ).configs[0].config
                        ),
                        indent=2
                    ),
                    height=200
                )
                generation_config_text = cc2.text_area(
                    "Generation Config",
                    json.dumps(
                        json.loads(
                            fts.ListConfigs(
                                ListConfigsRequest(
                                    type=ConfigType.GENERATION_CONFIG,
                                    model_id=current_models[mlflow_model_idx].id,
                                    adapter_id=adapter_id
                                )
                            ).configs[0].config
                        ),
                        indent=2
                    ),
                    height=200
                )

        start_job_button = st.button(
            "Start MLflow Evaluation Job",
            type="primary",
            use_container_width=True)

        if start_job_button:
            if not button_enabled:
                st.warning(
                    "Please complete the fields: Adapter, Dataset, Prompt, and Base Model before starting the job.",
                    icon=":material/error:")
            else:
                try:
                    model_adapter_combo: List[EvaluationJobModelCombination] = []
                    first_model = current_models[all_model_adapter_combinations[0]['mlflow_model_idx']]
                    prompt = current_prompts[mlflow_prompt_idx]
                    dataset = current_datasets[mlflow_dataset_idx]
                    for combo in all_model_adapter_combinations:
                        mlflow_model_idx, model_adapter = combo['mlflow_model_idx'], combo['model_adapter']
                        model = current_models[mlflow_model_idx]
                        if model_adapter == BASE_MODEL_ONLY_IDX:
                            adapter = AdapterMetadata()
                            adapter.id = BASE_MODEL_ONLY_ADAPTER_ID
                        else:
                            adapter = model_adapter

                        model_adapter_combo.append(EvaluationJobModelCombination(
                            base_model_id=model.id,
                            adapter_id=adapter.id))

                    # If there were any changes made to the generation or bnb config,
                    # add these new configs to the config store.
                    bnb_config_md: ConfigMetadata = fts.AddConfig(
                        AddConfigRequest(
                            type=ConfigType.BITSANDBYTES_CONFIG,
                            config=bnb_config_text,
                            description=model.huggingface_model_name
                        )
                    ).config
                    generation_config_md: ConfigMetadata = fts.AddConfig(
                        AddConfigRequest(
                            type=ConfigType.GENERATION_CONFIG,
                            config=generation_config_text,
                            description=model.huggingface_model_name
                        )
                    ).config

                    fts.StartEvaluationJob(
                        StartEvaluationJobRequest(
                            type=EvaluationJobType.MFLOW,
                            model_adapter_combinations=model_adapter_combo,
                            dataset_id=dataset.id,
                            prompt_id=prompt.id,
                            cpu=int(st.session_state['mlflow_cpu']),
                            gpu=gpu,
                            gpu_label_id=gpu_label_id,
                            memory=int(st.session_state['mlflow_memory']),
                            model_bnb_config_id=bnb_config_md.id,
                            adapter_bnb_config_id=bnb_config_md.id,
                            generation_config_id=generation_config_md.id,
                            selected_features=st.session_state['selected_features'],
                            eval_dataset_fraction=st.session_state['eval_dataset_fraction']
                        )
                    )
                    st.success("Created MLflow Job. Please go to **View MLflow Runs** tab!", icon=":material/check:")
                    st.toast("Created MLflow Job. Please go to **View MLflow Runs** tab!", icon=":material/check:")
                except Exception as e:
                    st.error(f"Failed to create MLflow Job: **{str(e)}**", icon=":material/error:")
                    st.toast(f"Failed to create MLflow Job: **{str(e)}**", icon=":material/error:")

with ccol2:
    st.info(
        """
        This page allows you to run MLflow evaluation jobs on your fine-tuned models and their corresponding adapters.
        The evaluation job will generate a detailed report on the performance of the adapter using a sample dataset.
        You can view the complete evaluation report on the **View MLflow Runs** page.
        """,
        icon=":material/info:"
    )
    st.write("\n")
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
