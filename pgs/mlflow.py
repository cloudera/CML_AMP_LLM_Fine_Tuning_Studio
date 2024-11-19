import streamlit as st
from ft.utils import get_env_variable, fetch_resource_usage_data, process_resource_usage_data, fetch_cml_site_config
from typing import List
from ft.api import *
from pgs.streamlit_utils import get_fine_tuning_studio_client, get_cml_client
import json
from ft.consts import IconPaths, DIVIDER_COLOR, DEFAULT_BNB_CONFIG, DEFAULT_GENERATIONAL_CONFIG

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

# New session state, list of dicts representing model selection
if 'mlflow_model_adapters' not in st.session_state:

    # Fill in a blank session state to start with no models
    # and no adapters selected
    st.session_state['mlflow_model_adapters'] = [
        {
            "base_model_id": None,
            "adapter_id": None
        }
    ]

if 'bnb_config' not in st.session_state:
    st.session_state['bnb_config'] = DEFAULT_BNB_CONFIG

if 'generation_config' not in st.session_state:
    st.session_state['generation_config'] = DEFAULT_GENERATIONAL_CONFIG


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
            index=st.session_state['mlflow_dataset_idx'],
            key="dataset_selectbox",
        )
        if 'prev_dataset_idx' not in st.session_state or st.session_state['prev_dataset_idx'] != mlflow_dataset_idx:
            st.session_state['mlflow_prompt_idx'] = None
            st.session_state['selected_features'] = []
            st.session_state['eval_dataset_fraction'] = 1.0
            st.session_state['mlflow_model_idx'] = None
            st.session_state['model_adapter'] = None
            st.session_state['prev_dataset_idx'] = mlflow_dataset_idx

        st.session_state['mlflow_dataset_idx'] = mlflow_dataset_idx

        if mlflow_dataset_idx is not None:
            st.session_state['mlflow_dataset_idx'] = mlflow_dataset_idx
            dataset = current_datasets[mlflow_dataset_idx]
            current_prompts = fts.get_prompts()
            current_prompts = list(filter(lambda x: x.dataset_id == dataset.id, current_prompts))
            if len(current_prompts) > 0:
                valid_index = 0
                if st.session_state['mlflow_prompt_idx'] is not None:
                    valid_index = min(len(current_prompts) - 1, max(0, st.session_state['mlflow_prompt_idx']))

                mlflow_prompt_idx = st.selectbox(
                    "Prompts",
                    range(len(current_prompts)),
                    format_func=lambda x: current_prompts[x].name,
                    index=valid_index,
                    key="prompt_selectbox",
                    help="Select the prompt to use with the selected dataset. This field is required."
                )
                st.session_state['mlflow_prompt_idx'] = mlflow_prompt_idx
                subcol1, subcol2 = st.columns(2)
                subcol1.caption("Prompt Template")
                subcol1.code(current_prompts[mlflow_prompt_idx].input_template)
                subcol2.caption("Completion Template")
                subcol2.code(current_prompts[mlflow_prompt_idx].completion_template)
            else:
                st.error(
                    "No prompts available. Please create a prompt template for the selected dataset to proceed with training.",
                    icon=":material/error:")

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
            selected_dataset_features = st.multiselect(
                "Choose Extra Columns That You Need In The Evaluation CSV",
                dataset_features,
                help="These extra columns will be included in the evaluation CSV along with the model input, expected output and the model output columns. Leave it blank for default behaviour.",
                key="selected_dataset_features",
                default=st.session_state['selected_features'],
            )

            st.session_state['selected_features'] = selected_dataset_features or []

        st.divider()
        st.caption("**Choose Models for Evaluation**")
        all_model_adapter_combinations = []
        # For every item in the session state for list of model adapter combos, see
        # what is selected and allow for changes dynamically.
        for i in range(len(st.session_state.mlflow_model_adapters)):

            with st.container(border=True):
                current_models = fts.get_models()
                current_model_id = st.session_state.mlflow_model_adapters[i].get("base_model_id", None)

                # Get the index from the list of models based on the model ID that is in
                # the session state for this model adapter pair.
                def get_model_index_based_on_id(id: str, models: List[ModelMetadata]) -> int:
                    for idx, model in enumerate(models):
                        if model.id == id:
                            return idx
                    return None

                # Define a callback function to update the session state accordingly. Note that
                # this code runs *before* a streamlit refresh.
                def update_model_selection():
                    current_model_index = st.session_state[f"current_model_index_{i}"]
                    selected_model = current_models[current_model_index]
                    if not selected_model.id == st.session_state.mlflow_model_adapters[i]["base_model_id"]:
                        st.session_state.mlflow_model_adapters[i]["adapter_id"] = None
                    st.session_state.mlflow_model_adapters[i]["base_model_id"] = selected_model.id

                # Get a list of all of the models
                # link the current selection based on the model ID if it's not none on this current pair
                mlflow_model_idx = st.selectbox(
                    "Base Model",
                    range(
                        len(current_models)),
                    format_func=lambda x: current_models[x].name,
                    index=get_model_index_based_on_id(
                        current_model_id,
                        current_models) if current_model_id is not None else None,
                    key=f"current_model_index_{i}",
                    on_change=update_model_selection)

                current_adapters = list(
                    filter(
                        lambda x: x.model_id == current_model_id,
                        fts.get_adapters())) if current_model_id is not None else []
                current_adapter_id = st.session_state.mlflow_model_adapters[i].get("adapter_id", None)

                def get_adapter_index_based_on_id(id: str, adapters: List[AdapterMetadata]) -> int:
                    for idx, adapter in enumerate(adapters):
                        if adapter.id == id:
                            return idx
                    return None

                model_adapter_idx = st.selectbox(
                    "(optional) Choose an Adapter",
                    range(
                        len(current_adapters)),
                    format_func=lambda x: current_adapters[x].name,
                    index=get_adapter_index_based_on_id(
                        current_adapter_id,
                        current_adapters) if current_adapter_id is not None else None,
                    key=f"current_adapter_index_{i}")
                if model_adapter_idx is not None:
                    selected_adapter: AdapterMetadata = current_adapters[model_adapter_idx]
                    st.session_state.mlflow_model_adapters[i]["adapter_id"] = selected_adapter.id

                # Add a remove button per combo button
                def remove_model_adapter_pair(idx: int):
                    del st.session_state.mlflow_model_adapters[idx]
                st.button(
                    label="Remove", key=f"remove_adapter_{i}", disabled=(
                        i == 0 and st.session_state.mlflow_model_adapters[i]["base_model_id"] is None and st.session_state.mlflow_model_adapters[i]["adapter_id"] is None) or len(
                        st.session_state.mlflow_model_adapters) <= 1, on_click=remove_model_adapter_pair, args=(
                        i,))

        # Add a model adapter pair
        def add_model_adapter_pair():
            st.session_state.mlflow_model_adapters.append(
                {
                    "base_model_id": None,
                    "adapter_id": None
                }
            )
        st.button(
            label="Add another model & adapter",
            on_click=add_model_adapter_pair,
            disabled=st.session_state.mlflow_model_adapters[-1]["base_model_id"] is None and st.session_state.mlflow_model_adapters[-1]["adapter_id"] is None
        )

        # Advanced options
        st.caption("**Advanced Options**")
        c1, c2 = st.columns([1, 1])
        with c1:
            mlflow_cpu = st.text_input("CPU(vCPU)", value="2", key="mlflow_cpu")
        with c2:
            mlflow_memory = st.text_input("Memory(GiB)", value="8", key="mlflow_memory")

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

        with st.expander("Configs"):
            cc1, cc2 = st.columns([1, 1])

            bnb_config_text = cc1.text_area(
                "Quantization Config",
                value=json.dumps(
                    st.session_state.bnb_config,
                    indent=2
                ),
                height=200
            )
            generation_config_text = cc2.text_area(
                "Generation Config",
                json.dumps(
                    st.session_state.generation_config,
                    indent=2
                ),
                height=200
            )
            st.session_state.bnb_config = json.loads(bnb_config_text)
            st.session_state.generation_config = json.loads(generation_config_text)

        start_job_button = st.button(
            "Start MLflow Evaluation Job",
            type="primary",
            use_container_width=True)

        if start_job_button:

            def empty_model_field_present():
                for pair in st.session_state.mlflow_model_adapters:
                    if pair["base_model_id"] is None and pair["adapter_id"] is None:
                        return True
                return False

            def repeated_model_adapter_pair():
                for i, pair1 in enumerate(st.session_state.mlflow_model_adapters):
                    for j, pair2 in enumerate(st.session_state.mlflow_model_adapters):
                        if (not i == j) and (pair1 == pair2):
                            return True
                return False

            def missing_dataset_selection():
                return mlflow_dataset_idx is None

            def missing_prompt_selection():
                return mlflow_prompt_idx is None

            if empty_model_field_present():
                st.warning(
                    "One of the selected model/adapter pairs is missing a base model. Make sure there is at least a base model in all model/adapter pairs to evaluate.",
                    icon=":material/error:")
            elif repeated_model_adapter_pair():
                st.warning(
                    "Two of the selected model/adapter pairs are the same model & adapter. Remove all duplicates from the list of model/adapters to evaluate.",
                    icon=":material/error:")
            elif missing_dataset_selection():
                st.warning(
                    "Please select a dataset to evaluate on.",
                    icon=":material/error:")
            elif missing_prompt_selection():
                st.warning(
                    "Please select a prompt template to evaluate on.",
                    icon=":material/error:")
            else:
                try:
                    model_adapter_combo: List[EvaluationJobModelCombination] = []
                    prompt = current_prompts[mlflow_prompt_idx]
                    dataset = current_datasets[mlflow_dataset_idx]
                    for combo in st.session_state.mlflow_model_adapters:
                        model_adapter_combo.append(EvaluationJobModelCombination(**combo))

                    # If there were any changes made to the generation or bnb config,
                    # add these new configs to the config store.
                    bnb_config_md: ConfigMetadata = fts.AddConfig(
                        AddConfigRequest(
                            type=ConfigType.BITSANDBYTES_CONFIG,
                            config=bnb_config_text
                        )
                    ).config
                    generation_config_md: ConfigMetadata = fts.AddConfig(
                        AddConfigRequest(
                            type=ConfigType.GENERATION_CONFIG,
                            config=generation_config_text
                        )
                    ).config

                    fts.StartEvaluationJob(
                        StartEvaluationJobRequest(
                            type=EvaluationJobType.MFLOW,
                            model_adapter_combinations=model_adapter_combo,
                            dataset_id=dataset.id,
                            prompt_id=prompt.id,
                            cpu=int(st.session_state.mlflow_cpu),
                            gpu=gpu,
                            gpu_label_id=int(st.session_state['ft_resource_gpu_label']),
                            memory=int(st.session_state.mlflow_memory),
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
