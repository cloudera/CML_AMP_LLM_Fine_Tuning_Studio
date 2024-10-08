from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor


class ListDatasetsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...


class ListDatasetsResponse(_message.Message):
    __slots__ = ("datasets",)
    DATASETS_FIELD_NUMBER: _ClassVar[int]
    datasets: _containers.RepeatedCompositeFieldContainer[DatasetMetadata]
    def __init__(self, datasets: _Optional[_Iterable[_Union[DatasetMetadata, _Mapping]]] = ...) -> None: ...


class GetDatasetRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...


class GetDatasetResponse(_message.Message):
    __slots__ = ("dataset",)
    DATASET_FIELD_NUMBER: _ClassVar[int]
    dataset: DatasetMetadata
    def __init__(self, dataset: _Optional[_Union[DatasetMetadata, _Mapping]] = ...) -> None: ...


class AddDatasetRequest(_message.Message):
    __slots__ = ("type", "huggingface_name", "location", "name")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    HUGGINGFACE_NAME_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    type: str
    huggingface_name: str
    location: str
    name: str

    def __init__(
        self,
        type: _Optional[str] = ...,
        huggingface_name: _Optional[str] = ...,
        location: _Optional[str] = ...,
        name: _Optional[str] = ...) -> None: ...


class AddDatasetResponse(_message.Message):
    __slots__ = ("dataset",)
    DATASET_FIELD_NUMBER: _ClassVar[int]
    dataset: DatasetMetadata
    def __init__(self, dataset: _Optional[_Union[DatasetMetadata, _Mapping]] = ...) -> None: ...


class RemoveDatasetRequest(_message.Message):
    __slots__ = ("id", "remove_prompts")
    ID_FIELD_NUMBER: _ClassVar[int]
    REMOVE_PROMPTS_FIELD_NUMBER: _ClassVar[int]
    id: str
    remove_prompts: bool
    def __init__(self, id: _Optional[str] = ..., remove_prompts: bool = ...) -> None: ...


class RemoveDatasetResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...


class GetDatasetSplitByAdapterRequest(_message.Message):
    __slots__ = ("adapter_id",)
    ADAPTER_ID_FIELD_NUMBER: _ClassVar[int]
    adapter_id: str
    def __init__(self, adapter_id: _Optional[str] = ...) -> None: ...


class GetDatasetSplitByAdapterResponse(_message.Message):
    __slots__ = ("response",)
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    response: GetDatasetSplitByAdapterMetadata
    def __init__(self, response: _Optional[_Union[GetDatasetSplitByAdapterMetadata, _Mapping]] = ...) -> None: ...


class GetDatasetSplitByAdapterMetadata(_message.Message):
    __slots__ = ("dataset_fraction", "train_test_split")
    DATASET_FRACTION_FIELD_NUMBER: _ClassVar[int]
    TRAIN_TEST_SPLIT_FIELD_NUMBER: _ClassVar[int]
    dataset_fraction: float
    train_test_split: float
    def __init__(self, dataset_fraction: _Optional[float] = ..., train_test_split: _Optional[float] = ...) -> None: ...


class ListModelsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...


class ListModelsResponse(_message.Message):
    __slots__ = ("models",)
    MODELS_FIELD_NUMBER: _ClassVar[int]
    models: _containers.RepeatedCompositeFieldContainer[ModelMetadata]
    def __init__(self, models: _Optional[_Iterable[_Union[ModelMetadata, _Mapping]]] = ...) -> None: ...


class GetModelRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...


class GetModelResponse(_message.Message):
    __slots__ = ("model",)
    MODEL_FIELD_NUMBER: _ClassVar[int]
    model: ModelMetadata
    def __init__(self, model: _Optional[_Union[ModelMetadata, _Mapping]] = ...) -> None: ...


class AddModelRequest(_message.Message):
    __slots__ = ("type", "huggingface_name", "model_registry_id")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    HUGGINGFACE_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_REGISTRY_ID_FIELD_NUMBER: _ClassVar[int]
    type: str
    huggingface_name: str
    model_registry_id: str
    def __init__(self, type: _Optional[str] = ..., huggingface_name: _Optional[str]
                 = ..., model_registry_id: _Optional[str] = ...) -> None: ...


class AddModelResponse(_message.Message):
    __slots__ = ("model",)
    MODEL_FIELD_NUMBER: _ClassVar[int]
    model: ModelMetadata
    def __init__(self, model: _Optional[_Union[ModelMetadata, _Mapping]] = ...) -> None: ...


class ExportModelRequest(_message.Message):
    __slots__ = ("type", "model_id", "adapter_id", "model_name", "artifact_location", "model_description")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_LOCATION_FIELD_NUMBER: _ClassVar[int]
    MODEL_DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    type: str
    model_id: str
    adapter_id: str
    model_name: str
    artifact_location: str
    model_description: str

    def __init__(
        self,
        type: _Optional[str] = ...,
        model_id: _Optional[str] = ...,
        adapter_id: _Optional[str] = ...,
        model_name: _Optional[str] = ...,
        artifact_location: _Optional[str] = ...,
        model_description: _Optional[str] = ...) -> None: ...


class ExportModelResponse(_message.Message):
    __slots__ = ("model",)
    MODEL_FIELD_NUMBER: _ClassVar[int]
    model: ModelMetadata
    def __init__(self, model: _Optional[_Union[ModelMetadata, _Mapping]] = ...) -> None: ...


class RemoveModelRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...


class RemoveModelResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...


class ListAdaptersRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...


class ListAdaptersResponse(_message.Message):
    __slots__ = ("adapters",)
    ADAPTERS_FIELD_NUMBER: _ClassVar[int]
    adapters: _containers.RepeatedCompositeFieldContainer[AdapterMetadata]
    def __init__(self, adapters: _Optional[_Iterable[_Union[AdapterMetadata, _Mapping]]] = ...) -> None: ...


class GetAdapterRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...


class GetAdapterResponse(_message.Message):
    __slots__ = ("adapter",)
    ADAPTER_FIELD_NUMBER: _ClassVar[int]
    adapter: AdapterMetadata
    def __init__(self, adapter: _Optional[_Union[AdapterMetadata, _Mapping]] = ...) -> None: ...


class AddAdapterRequest(_message.Message):
    __slots__ = ("type", "name", "model_id", "location", "huggingface_name", "fine_tuning_job_id", "prompt_id")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    HUGGINGFACE_NAME_FIELD_NUMBER: _ClassVar[int]
    FINE_TUNING_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_ID_FIELD_NUMBER: _ClassVar[int]
    type: str
    name: str
    model_id: str
    location: str
    huggingface_name: str
    fine_tuning_job_id: str
    prompt_id: str

    def __init__(
        self,
        type: _Optional[str] = ...,
        name: _Optional[str] = ...,
        model_id: _Optional[str] = ...,
        location: _Optional[str] = ...,
        huggingface_name: _Optional[str] = ...,
        fine_tuning_job_id: _Optional[str] = ...,
        prompt_id: _Optional[str] = ...) -> None: ...


class AddAdapterResponse(_message.Message):
    __slots__ = ("adapter",)
    ADAPTER_FIELD_NUMBER: _ClassVar[int]
    adapter: AdapterMetadata
    def __init__(self, adapter: _Optional[_Union[AdapterMetadata, _Mapping]] = ...) -> None: ...


class RemoveAdapterRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...


class RemoveAdapterResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...


class ListPromptsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...


class ListPromptsResponse(_message.Message):
    __slots__ = ("prompts",)
    PROMPTS_FIELD_NUMBER: _ClassVar[int]
    prompts: _containers.RepeatedCompositeFieldContainer[PromptMetadata]
    def __init__(self, prompts: _Optional[_Iterable[_Union[PromptMetadata, _Mapping]]] = ...) -> None: ...


class GetPromptRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...


class GetPromptResponse(_message.Message):
    __slots__ = ("prompt",)
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    prompt: PromptMetadata
    def __init__(self, prompt: _Optional[_Union[PromptMetadata, _Mapping]] = ...) -> None: ...


class AddPromptRequest(_message.Message):
    __slots__ = ("prompt",)
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    prompt: PromptMetadata
    def __init__(self, prompt: _Optional[_Union[PromptMetadata, _Mapping]] = ...) -> None: ...


class AddPromptResponse(_message.Message):
    __slots__ = ("prompt",)
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    prompt: PromptMetadata
    def __init__(self, prompt: _Optional[_Union[PromptMetadata, _Mapping]] = ...) -> None: ...


class RemovePromptRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...


class RemovePromptResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...


class ListFineTuningJobsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...


class ListFineTuningJobsResponse(_message.Message):
    __slots__ = ("fine_tuning_jobs",)
    FINE_TUNING_JOBS_FIELD_NUMBER: _ClassVar[int]
    fine_tuning_jobs: _containers.RepeatedCompositeFieldContainer[FineTuningJobMetadata]
    def __init__(
        self, fine_tuning_jobs: _Optional[_Iterable[_Union[FineTuningJobMetadata, _Mapping]]] = ...) -> None: ...


class GetFineTuningJobRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...


class GetFineTuningJobResponse(_message.Message):
    __slots__ = ("fine_tuning_job",)
    FINE_TUNING_JOB_FIELD_NUMBER: _ClassVar[int]
    fine_tuning_job: FineTuningJobMetadata
    def __init__(self, fine_tuning_job: _Optional[_Union[FineTuningJobMetadata, _Mapping]] = ...) -> None: ...


class StartFineTuningJobRequest(_message.Message):
    __slots__ = (
        "adapter_name",
        "base_model_id",
        "dataset_id",
        "prompt_id",
        "num_workers",
        "auto_add_adapter",
        "num_epochs",
        "learning_rate",
        "cpu",
        "gpu",
        "memory",
        "train_test_split",
        "model_bnb_config_id",
        "adapter_bnb_config_id",
        "training_arguments_config_id",
        "lora_config_id",
        "output_dir",
        "dataset_fraction",
        "user_script",
        "user_config_id",
        "user_config",
        "framework_type",
        "axolotl_config_id",
        "gpu_label_id")
    ADAPTER_NAME_FIELD_NUMBER: _ClassVar[int]
    BASE_MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_ID_FIELD_NUMBER: _ClassVar[int]
    NUM_WORKERS_FIELD_NUMBER: _ClassVar[int]
    AUTO_ADD_ADAPTER_FIELD_NUMBER: _ClassVar[int]
    NUM_EPOCHS_FIELD_NUMBER: _ClassVar[int]
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    CPU_FIELD_NUMBER: _ClassVar[int]
    GPU_FIELD_NUMBER: _ClassVar[int]
    MEMORY_FIELD_NUMBER: _ClassVar[int]
    TRAIN_TEST_SPLIT_FIELD_NUMBER: _ClassVar[int]
    MODEL_BNB_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_BNB_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    TRAINING_ARGUMENTS_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    LORA_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_DIR_FIELD_NUMBER: _ClassVar[int]
    DATASET_FRACTION_FIELD_NUMBER: _ClassVar[int]
    USER_SCRIPT_FIELD_NUMBER: _ClassVar[int]
    USER_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    USER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    FRAMEWORK_TYPE_FIELD_NUMBER: _ClassVar[int]
    AXOLOTL_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    GPU_LABEL_ID_FIELD_NUMBER: _ClassVar[int]
    adapter_name: str
    base_model_id: str
    dataset_id: str
    prompt_id: str
    num_workers: int
    auto_add_adapter: bool
    num_epochs: int
    learning_rate: float
    cpu: int
    gpu: int
    memory: int
    train_test_split: float
    model_bnb_config_id: str
    adapter_bnb_config_id: str
    training_arguments_config_id: str
    lora_config_id: str
    output_dir: str
    dataset_fraction: float
    user_script: str
    user_config_id: str
    user_config: str
    framework_type: str
    axolotl_config_id: str
    gpu_label_id: int

    def __init__(
        self,
        adapter_name: _Optional[str] = ...,
        base_model_id: _Optional[str] = ...,
        dataset_id: _Optional[str] = ...,
        prompt_id: _Optional[str] = ...,
        num_workers: _Optional[int] = ...,
        auto_add_adapter: bool = ...,
        num_epochs: _Optional[int] = ...,
        learning_rate: _Optional[float] = ...,
        cpu: _Optional[int] = ...,
        gpu: _Optional[int] = ...,
        memory: _Optional[int] = ...,
        train_test_split: _Optional[float] = ...,
        model_bnb_config_id: _Optional[str] = ...,
        adapter_bnb_config_id: _Optional[str] = ...,
        training_arguments_config_id: _Optional[str] = ...,
        lora_config_id: _Optional[str] = ...,
        output_dir: _Optional[str] = ...,
        dataset_fraction: _Optional[float] = ...,
        user_script: _Optional[str] = ...,
        user_config_id: _Optional[str] = ...,
        user_config: _Optional[str] = ...,
        framework_type: _Optional[str] = ...,
        axolotl_config_id: _Optional[str] = ...,
        gpu_label_id: _Optional[int] = ...) -> None: ...


class StartFineTuningJobResponse(_message.Message):
    __slots__ = ("fine_tuning_job",)
    FINE_TUNING_JOB_FIELD_NUMBER: _ClassVar[int]
    fine_tuning_job: FineTuningJobMetadata
    def __init__(self, fine_tuning_job: _Optional[_Union[FineTuningJobMetadata, _Mapping]] = ...) -> None: ...


class RemoveFineTuningJobRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...


class RemoveFineTuningJobResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...


class ListEvaluationJobsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...


class ListEvaluationJobsResponse(_message.Message):
    __slots__ = ("evaluation_jobs",)
    EVALUATION_JOBS_FIELD_NUMBER: _ClassVar[int]
    evaluation_jobs: _containers.RepeatedCompositeFieldContainer[EvaluationJobMetadata]
    def __init__(
        self, evaluation_jobs: _Optional[_Iterable[_Union[EvaluationJobMetadata, _Mapping]]] = ...) -> None: ...


class GetEvaluationJobRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...


class GetEvaluationJobResponse(_message.Message):
    __slots__ = ("evaluation_job",)
    EVALUATION_JOB_FIELD_NUMBER: _ClassVar[int]
    evaluation_job: EvaluationJobMetadata
    def __init__(self, evaluation_job: _Optional[_Union[EvaluationJobMetadata, _Mapping]] = ...) -> None: ...


class EvaluationJobModelCombination(_message.Message):
    __slots__ = ("base_model_id", "adapter_id")
    BASE_MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_ID_FIELD_NUMBER: _ClassVar[int]
    base_model_id: str
    adapter_id: str
    def __init__(self, base_model_id: _Optional[str] = ..., adapter_id: _Optional[str] = ...) -> None: ...


class StartEvaluationJobRequest(_message.Message):
    __slots__ = (
        "type",
        "model_adapter_combinations",
        "dataset_id",
        "cpu",
        "gpu",
        "memory",
        "model_bnb_config_id",
        "adapter_bnb_config_id",
        "generation_config_id",
        "prompt_id",
        "gpu_label_id",
        "selected_features",
        "eval_dataset_fraction")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ADAPTER_COMBINATIONS_FIELD_NUMBER: _ClassVar[int]
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    CPU_FIELD_NUMBER: _ClassVar[int]
    GPU_FIELD_NUMBER: _ClassVar[int]
    MEMORY_FIELD_NUMBER: _ClassVar[int]
    MODEL_BNB_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_BNB_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    GENERATION_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_ID_FIELD_NUMBER: _ClassVar[int]
    GPU_LABEL_ID_FIELD_NUMBER: _ClassVar[int]
    SELECTED_FEATURES_FIELD_NUMBER: _ClassVar[int]
    EVAL_DATASET_FRACTION_FIELD_NUMBER: _ClassVar[int]
    type: str
    model_adapter_combinations: _containers.RepeatedCompositeFieldContainer[EvaluationJobModelCombination]
    dataset_id: str
    cpu: int
    gpu: int
    memory: int
    model_bnb_config_id: str
    adapter_bnb_config_id: str
    generation_config_id: str
    prompt_id: str
    gpu_label_id: int
    selected_features: _containers.RepeatedScalarFieldContainer[str]
    eval_dataset_fraction: float

    def __init__(self,
                 type: _Optional[str] = ...,
                 model_adapter_combinations: _Optional[_Iterable[_Union[EvaluationJobModelCombination,
                                                                        _Mapping]]] = ...,
                 dataset_id: _Optional[str] = ...,
                 cpu: _Optional[int] = ...,
                 gpu: _Optional[int] = ...,
                 memory: _Optional[int] = ...,
                 model_bnb_config_id: _Optional[str] = ...,
                 adapter_bnb_config_id: _Optional[str] = ...,
                 generation_config_id: _Optional[str] = ...,
                 prompt_id: _Optional[str] = ...,
                 gpu_label_id: _Optional[int] = ...,
                 selected_features: _Optional[_Iterable[str]] = ...,
                 eval_dataset_fraction: _Optional[float] = ...) -> None: ...


class StartEvaluationJobResponse(_message.Message):
    __slots__ = ("evaluation_job",)
    EVALUATION_JOB_FIELD_NUMBER: _ClassVar[int]
    evaluation_job: EvaluationJobMetadata
    def __init__(self, evaluation_job: _Optional[_Union[EvaluationJobMetadata, _Mapping]] = ...) -> None: ...


class RemoveEvaluationJobRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...


class RemoveEvaluationJobResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...


class ListConfigsRequest(_message.Message):
    __slots__ = ("type", "model_id", "adapter_id", "fine_tuning_job_id")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_ID_FIELD_NUMBER: _ClassVar[int]
    FINE_TUNING_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    type: str
    model_id: str
    adapter_id: str
    fine_tuning_job_id: str

    def __init__(
        self,
        type: _Optional[str] = ...,
        model_id: _Optional[str] = ...,
        adapter_id: _Optional[str] = ...,
        fine_tuning_job_id: _Optional[str] = ...) -> None: ...


class ListConfigsResponse(_message.Message):
    __slots__ = ("configs",)
    CONFIGS_FIELD_NUMBER: _ClassVar[int]
    configs: _containers.RepeatedCompositeFieldContainer[ConfigMetadata]
    def __init__(self, configs: _Optional[_Iterable[_Union[ConfigMetadata, _Mapping]]] = ...) -> None: ...


class GetConfigRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...


class GetConfigResponse(_message.Message):
    __slots__ = ("config",)
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    config: ConfigMetadata
    def __init__(self, config: _Optional[_Union[ConfigMetadata, _Mapping]] = ...) -> None: ...


class AddConfigRequest(_message.Message):
    __slots__ = ("type", "config", "description")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    type: str
    config: str
    description: str
    def __init__(self, type: _Optional[str] = ..., config: _Optional[str]
                 = ..., description: _Optional[str] = ...) -> None: ...


class AddConfigResponse(_message.Message):
    __slots__ = ("config",)
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    config: ConfigMetadata
    def __init__(self, config: _Optional[_Union[ConfigMetadata, _Mapping]] = ...) -> None: ...


class RemoveConfigRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...


class RemoveConfigResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...


class DatasetMetadata(_message.Message):
    __slots__ = ("id", "type", "name", "description", "huggingface_name", "location", "features")
    ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    HUGGINGFACE_NAME_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    id: str
    type: str
    name: str
    description: str
    huggingface_name: str
    location: str
    features: str

    def __init__(
        self,
        id: _Optional[str] = ...,
        type: _Optional[str] = ...,
        name: _Optional[str] = ...,
        description: _Optional[str] = ...,
        huggingface_name: _Optional[str] = ...,
        location: _Optional[str] = ...,
        features: _Optional[str] = ...) -> None: ...


class ModelMetadata(_message.Message):
    __slots__ = (
        "id",
        "type",
        "framework",
        "name",
        "huggingface_model_name",
        "location",
        "cml_registered_model_id",
        "mlflow_experiment_id",
        "mlflow_run_id")
    ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    FRAMEWORK_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    HUGGINGFACE_MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    CML_REGISTERED_MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MLFLOW_EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    MLFLOW_RUN_ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    type: str
    framework: str
    name: str
    huggingface_model_name: str
    location: str
    cml_registered_model_id: str
    mlflow_experiment_id: str
    mlflow_run_id: str

    def __init__(
        self,
        id: _Optional[str] = ...,
        type: _Optional[str] = ...,
        framework: _Optional[str] = ...,
        name: _Optional[str] = ...,
        huggingface_model_name: _Optional[str] = ...,
        location: _Optional[str] = ...,
        cml_registered_model_id: _Optional[str] = ...,
        mlflow_experiment_id: _Optional[str] = ...,
        mlflow_run_id: _Optional[str] = ...) -> None: ...


class AdapterMetadata(_message.Message):
    __slots__ = (
        "id",
        "type",
        "name",
        "model_id",
        "location",
        "huggingface_name",
        "fine_tuning_job_id",
        "prompt_id",
        "cml_registered_model_id",
        "mlflow_experiment_id",
        "mlflow_run_id")
    ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    HUGGINGFACE_NAME_FIELD_NUMBER: _ClassVar[int]
    FINE_TUNING_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_ID_FIELD_NUMBER: _ClassVar[int]
    CML_REGISTERED_MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MLFLOW_EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    MLFLOW_RUN_ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    type: str
    name: str
    model_id: str
    location: str
    huggingface_name: str
    fine_tuning_job_id: str
    prompt_id: str
    cml_registered_model_id: str
    mlflow_experiment_id: str
    mlflow_run_id: str

    def __init__(
        self,
        id: _Optional[str] = ...,
        type: _Optional[str] = ...,
        name: _Optional[str] = ...,
        model_id: _Optional[str] = ...,
        location: _Optional[str] = ...,
        huggingface_name: _Optional[str] = ...,
        fine_tuning_job_id: _Optional[str] = ...,
        prompt_id: _Optional[str] = ...,
        cml_registered_model_id: _Optional[str] = ...,
        mlflow_experiment_id: _Optional[str] = ...,
        mlflow_run_id: _Optional[str] = ...) -> None: ...


class PromptMetadata(_message.Message):
    __slots__ = ("id", "type", "name", "dataset_id", "prompt_template", "input_template", "completion_template")
    ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    INPUT_TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    id: str
    type: str
    name: str
    dataset_id: str
    prompt_template: str
    input_template: str
    completion_template: str

    def __init__(
        self,
        id: _Optional[str] = ...,
        type: _Optional[str] = ...,
        name: _Optional[str] = ...,
        dataset_id: _Optional[str] = ...,
        prompt_template: _Optional[str] = ...,
        input_template: _Optional[str] = ...,
        completion_template: _Optional[str] = ...) -> None: ...


class FineTuningJobMetadata(_message.Message):
    __slots__ = (
        "id",
        "base_model_id",
        "dataset_id",
        "prompt_id",
        "num_workers",
        "cml_job_id",
        "adapter_id",
        "num_cpu",
        "num_memory",
        "num_gpu",
        "num_epochs",
        "learning_rate",
        "out_dir",
        "training_arguments_config_id",
        "model_bnb_config_id",
        "adapter_bnb_config_id",
        "lora_config_id",
        "dataset_fraction",
        "train_test_split",
        "user_script",
        "user_config_id",
        "user_config",
        "framework_type",
        "axolotl_config_id",
        "gpu_label_id",
        "adapter_name")
    ID_FIELD_NUMBER: _ClassVar[int]
    BASE_MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_ID_FIELD_NUMBER: _ClassVar[int]
    NUM_WORKERS_FIELD_NUMBER: _ClassVar[int]
    CML_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_ID_FIELD_NUMBER: _ClassVar[int]
    NUM_CPU_FIELD_NUMBER: _ClassVar[int]
    NUM_MEMORY_FIELD_NUMBER: _ClassVar[int]
    NUM_GPU_FIELD_NUMBER: _ClassVar[int]
    NUM_EPOCHS_FIELD_NUMBER: _ClassVar[int]
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    OUT_DIR_FIELD_NUMBER: _ClassVar[int]
    TRAINING_ARGUMENTS_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_BNB_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_BNB_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    LORA_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    DATASET_FRACTION_FIELD_NUMBER: _ClassVar[int]
    TRAIN_TEST_SPLIT_FIELD_NUMBER: _ClassVar[int]
    USER_SCRIPT_FIELD_NUMBER: _ClassVar[int]
    USER_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    USER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    FRAMEWORK_TYPE_FIELD_NUMBER: _ClassVar[int]
    AXOLOTL_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    GPU_LABEL_ID_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_NAME_FIELD_NUMBER: _ClassVar[int]
    id: str
    base_model_id: str
    dataset_id: str
    prompt_id: str
    num_workers: int
    cml_job_id: str
    adapter_id: str
    num_cpu: int
    num_memory: int
    num_gpu: int
    num_epochs: int
    learning_rate: float
    out_dir: str
    training_arguments_config_id: str
    model_bnb_config_id: str
    adapter_bnb_config_id: str
    lora_config_id: str
    dataset_fraction: float
    train_test_split: float
    user_script: str
    user_config_id: str
    user_config: str
    framework_type: str
    axolotl_config_id: str
    gpu_label_id: int
    adapter_name: str

    def __init__(
        self,
        id: _Optional[str] = ...,
        base_model_id: _Optional[str] = ...,
        dataset_id: _Optional[str] = ...,
        prompt_id: _Optional[str] = ...,
        num_workers: _Optional[int] = ...,
        cml_job_id: _Optional[str] = ...,
        adapter_id: _Optional[str] = ...,
        num_cpu: _Optional[int] = ...,
        num_memory: _Optional[int] = ...,
        num_gpu: _Optional[int] = ...,
        num_epochs: _Optional[int] = ...,
        learning_rate: _Optional[float] = ...,
        out_dir: _Optional[str] = ...,
        training_arguments_config_id: _Optional[str] = ...,
        model_bnb_config_id: _Optional[str] = ...,
        adapter_bnb_config_id: _Optional[str] = ...,
        lora_config_id: _Optional[str] = ...,
        dataset_fraction: _Optional[float] = ...,
        train_test_split: _Optional[float] = ...,
        user_script: _Optional[str] = ...,
        user_config_id: _Optional[str] = ...,
        user_config: _Optional[str] = ...,
        framework_type: _Optional[str] = ...,
        axolotl_config_id: _Optional[str] = ...,
        gpu_label_id: _Optional[int] = ...,
        adapter_name: _Optional[str] = ...) -> None: ...


class ConfigMetadata(_message.Message):
    __slots__ = ("id", "description", "type", "config", "model_family", "is_default")
    ID_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    MODEL_FAMILY_FIELD_NUMBER: _ClassVar[int]
    IS_DEFAULT_FIELD_NUMBER: _ClassVar[int]
    id: str
    description: str
    type: str
    config: str
    model_family: str
    is_default: int

    def __init__(
        self,
        id: _Optional[str] = ...,
        description: _Optional[str] = ...,
        type: _Optional[str] = ...,
        config: _Optional[str] = ...,
        model_family: _Optional[str] = ...,
        is_default: _Optional[int] = ...) -> None: ...


class EvaluationJobMetadata(_message.Message):
    __slots__ = (
        "id",
        "cml_job_id",
        "base_model_id",
        "dataset_id",
        "num_workers",
        "adapter_id",
        "num_cpu",
        "num_memory",
        "num_gpu",
        "evaluation_dir",
        "model_bnb_config_id",
        "adapter_bnb_config_id",
        "generation_config_id",
        "type",
        "prompt_id",
        "parent_job_id")
    ID_FIELD_NUMBER: _ClassVar[int]
    CML_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    BASE_MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    NUM_WORKERS_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_ID_FIELD_NUMBER: _ClassVar[int]
    NUM_CPU_FIELD_NUMBER: _ClassVar[int]
    NUM_MEMORY_FIELD_NUMBER: _ClassVar[int]
    NUM_GPU_FIELD_NUMBER: _ClassVar[int]
    EVALUATION_DIR_FIELD_NUMBER: _ClassVar[int]
    MODEL_BNB_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_BNB_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    GENERATION_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    PROMPT_ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    cml_job_id: str
    base_model_id: str
    dataset_id: str
    num_workers: int
    adapter_id: str
    num_cpu: int
    num_memory: int
    num_gpu: int
    evaluation_dir: str
    model_bnb_config_id: str
    adapter_bnb_config_id: str
    generation_config_id: str
    type: str
    prompt_id: str
    parent_job_id: str

    def __init__(
        self,
        id: _Optional[str] = ...,
        cml_job_id: _Optional[str] = ...,
        base_model_id: _Optional[str] = ...,
        dataset_id: _Optional[str] = ...,
        num_workers: _Optional[int] = ...,
        adapter_id: _Optional[str] = ...,
        num_cpu: _Optional[int] = ...,
        num_memory: _Optional[int] = ...,
        num_gpu: _Optional[int] = ...,
        evaluation_dir: _Optional[str] = ...,
        model_bnb_config_id: _Optional[str] = ...,
        adapter_bnb_config_id: _Optional[str] = ...,
        generation_config_id: _Optional[str] = ...,
        type: _Optional[str] = ...,
        prompt_id: _Optional[str] = ...,
        parent_job_id: _Optional[str] = ...) -> None: ...
