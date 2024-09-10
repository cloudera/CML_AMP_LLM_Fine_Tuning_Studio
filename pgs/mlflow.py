import streamlit as st
import os
from ft.utils import get_env_variable, fetch_resource_usage_data, process_resource_usage_data, fetch_cml_site_config
from typing import List
from ft.api import *
from pgs.streamlit_utils import get_fine_tuning_studio_client, get_cml_client
import json
from ft.consts import IconPaths, DIVIDER_COLOR

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
    st.session_state['eval_dataset_fraction'] = 1
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

        CURRENT_MODEL = None

        current_models = fts.get_models()
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

            model_adapter_idx = st.selectbox(
                "Choose an Adapter",
                range(len(model_adapters)),
                format_func=lambda x: model_adapters[x].name,
                index=None
            )

            if len(model_adapters) == 0:
                st.error(
                    "No adapters available. Please create a fine tuning job for the selected base model to create an adapter.",
                    icon=":material/error:")

            if model_adapter_idx is not None:
                model_adapter = model_adapters[model_adapter_idx]

            # Container for dataset and prompt selection
            col1, col2 = st.columns(2)

        current_datasets = fts.get_datasets()
        dataset_idx = st.selectbox(
            "Datasets",
            range(len(current_datasets)),
            format_func=lambda x: current_datasets[x].name,
            index=None
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
                key="prompt_selectbox",
                help="Select the prompt to use with the selected dataset. This field is required."
            )

            if len(current_prompts) == 0:
                st.error(
                    "No prompts available. Please create a prompt template for the selected dataset to proceed with training.",
                    icon=":material/error:")

            if prompt_idx is not None:
                subcol1, subcol2 = st.columns(2)
                subcol1.caption("Prompt Template")
                subcol1.code(current_prompts[prompt_idx].input_template)
                subcol2.caption("Completion Template")
                subcol2.code(current_prompts[prompt_idx].completion_template)

            eval_dataset_fraction = st.slider(
                "Evaluation Dataset Fraction",
                min_value=0.0,
                max_value=1.0,
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
                default=st.session_state.selected_features,
            )

            st.session_state.selected_features = selected_features or []

        # Advanced options
        st.caption("**Advanced Options**")
        c1, c2 = st.columns([1, 1])
        with c1:
            cpu = st.text_input("CPU(vCPU)", value="2", key="cpu")
        with c2:
            memory = st.text_input("Memory(GiB)", value="8", key="memory")

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

        button_enabled = dataset_idx is not None and model_idx is not None and model_adapter_idx is not None and prompt_idx is not None

        if button_enabled:
            with st.expander("Configs"):
                cc1, cc2 = st.columns([1, 1])

                # Extract out a BnB config and a generation config that will be used for
                # this specific mlflow evaluation run. Right now there is no selection logic on
                # these configs for a specific model type, but there may be in the future. For now,
                # just use the first selected configuration for each.
                bnb_config_text = cc1.text_area(
                    "Quantization Config",
                    json.dumps(
                        json.loads(
                            fts.ListConfigs(
                                ListConfigsRequest(
                                    type=ConfigType.BITSANDBYTES_CONFIG,
                                    model_id=current_models[model_idx].id,
                                    adapter_id=model_adapters[model_adapter_idx].id
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
                                    model_id=current_models[model_idx].id,
                                    adapter_id=model_adapters[model_adapter_idx].id
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
                    model = current_models[model_idx]
                    dataset = current_datasets[dataset_idx]
                    adapter = model_adapters[model_adapter_idx]
                    prompt = current_prompts[prompt_idx]

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
                            adapter_id=adapter.id,
                            base_model_id=model.id,
                            dataset_id=dataset.id,
                            prompt_id=prompt.id,
                            cpu=int(cpu),
                            gpu=gpu,
                            gpu_label_id=int(gpu_label_id),
                            memory=int(memory),
                            model_bnb_config_id=bnb_config_md.id,
                            adapter_bnb_config_id=bnb_config_md.id,
                            generation_config_id=generation_config_md.id,
                            selected_features=st.session_state.selected_features,
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
