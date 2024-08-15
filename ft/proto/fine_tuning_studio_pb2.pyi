from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor


class DatasetType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DATASET_TYPE_HUGGINGFACE: _ClassVar[DatasetType]
    DATASET_TYPE_PROJECT: _ClassVar[DatasetType]


class ModelType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MODEL_TYPE_HUGGINGFACE: _ClassVar[ModelType]
    MODEL_TYPE_PROJECT: _ClassVar[ModelType]
    MODEL_TYPE_MODEL_REGISTRY: _ClassVar[ModelType]


class ModelFrameworkType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MODEL_FRAMEWORK_TYPE_PYTORCH: _ClassVar[ModelFrameworkType]
    MODEL_FRAMEWORK_TYPE_TENSORFLOW: _ClassVar[ModelFrameworkType]
    MODEL_FRAMEWORK_TYPE_ONNX: _ClassVar[ModelFrameworkType]


class AdapterType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ADAPTER_TYPE_PROJECT: _ClassVar[AdapterType]
    ADAPTER_TYPE_HUGGINGFACE: _ClassVar[AdapterType]
    ADAPTER_TYPE_MODEL_REGISTRY: _ClassVar[AdapterType]


class JobStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    JOB_STATUS_SCHEDULED: _ClassVar[JobStatus]
    JOB_STATUS_RUNNING: _ClassVar[JobStatus]
    JOB_STATUS_SUCCESS: _ClassVar[JobStatus]
    JOB_STATUS_FAILURE: _ClassVar[JobStatus]


class PromptType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PROMPT_TYPE_IN_PLACE: _ClassVar[PromptType]


class EvaluationJobType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EVALUATION_JOB_TYPE_MLFLOW: _ClassVar[EvaluationJobType]


DATASET_TYPE_HUGGINGFACE: DatasetType
DATASET_TYPE_PROJECT: DatasetType
MODEL_TYPE_HUGGINGFACE: ModelType
MODEL_TYPE_PROJECT: ModelType
MODEL_TYPE_MODEL_REGISTRY: ModelType
MODEL_FRAMEWORK_TYPE_PYTORCH: ModelFrameworkType
MODEL_FRAMEWORK_TYPE_TENSORFLOW: ModelFrameworkType
MODEL_FRAMEWORK_TYPE_ONNX: ModelFrameworkType
ADAPTER_TYPE_PROJECT: AdapterType
ADAPTER_TYPE_HUGGINGFACE: AdapterType
ADAPTER_TYPE_MODEL_REGISTRY: AdapterType
JOB_STATUS_SCHEDULED: JobStatus
JOB_STATUS_RUNNING: JobStatus
JOB_STATUS_SUCCESS: JobStatus
JOB_STATUS_FAILURE: JobStatus
PROMPT_TYPE_IN_PLACE: PromptType
EVALUATION_JOB_TYPE_MLFLOW: EvaluationJobType


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
    __slots__ = ("type", "huggingface_name", "location")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    HUGGINGFACE_NAME_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    type: DatasetType
    huggingface_name: str
    location: str
    def __init__(self, type: _Optional[_Union[DatasetType, str]] = ...,
                 huggingface_name: _Optional[str] = ..., location: _Optional[str] = ...) -> None: ...


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
    type: ModelType
    huggingface_name: str
    model_registry_id: str

    def __init__(self,
                 type: _Optional[_Union[ModelType,
                                        str]] = ...,
                 huggingface_name: _Optional[str] = ...,
                 model_registry_id: _Optional[str] = ...) -> None: ...


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
    type: ModelType
    model_id: str
    adapter_id: str
    model_name: str
    artifact_location: str
    model_description: str

    def __init__(self,
                 type: _Optional[_Union[ModelType,
                                        str]] = ...,
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
    __slots__ = ("adapter",)
    ADAPTER_FIELD_NUMBER: _ClassVar[int]
    adapter: AdapterMetadata
    def __init__(self, adapter: _Optional[_Union[AdapterMetadata, _Mapping]] = ...) -> None: ...


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
        "bits_and_bytes_config",
        "auto_add_adapter",
        "num_epochs",
        "learning_rate",
        "cpu",
        "gpu",
        "memory",
        "train_test_split")
    ADAPTER_NAME_FIELD_NUMBER: _ClassVar[int]
    BASE_MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_ID_FIELD_NUMBER: _ClassVar[int]
    NUM_WORKERS_FIELD_NUMBER: _ClassVar[int]
    BITS_AND_BYTES_CONFIG_FIELD_NUMBER: _ClassVar[int]
    AUTO_ADD_ADAPTER_FIELD_NUMBER: _ClassVar[int]
    NUM_EPOCHS_FIELD_NUMBER: _ClassVar[int]
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    CPU_FIELD_NUMBER: _ClassVar[int]
    GPU_FIELD_NUMBER: _ClassVar[int]
    MEMORY_FIELD_NUMBER: _ClassVar[int]
    TRAIN_TEST_SPLIT_FIELD_NUMBER: _ClassVar[int]
    adapter_name: str
    base_model_id: str
    dataset_id: str
    prompt_id: str
    num_workers: int
    bits_and_bytes_config: BnbConfig
    auto_add_adapter: bool
    num_epochs: int
    learning_rate: float
    cpu: int
    gpu: int
    memory: int
    train_test_split: float

    def __init__(self,
                 adapter_name: _Optional[str] = ...,
                 base_model_id: _Optional[str] = ...,
                 dataset_id: _Optional[str] = ...,
                 prompt_id: _Optional[str] = ...,
                 num_workers: _Optional[int] = ...,
                 bits_and_bytes_config: _Optional[_Union[BnbConfig,
                                                         _Mapping]] = ...,
                 auto_add_adapter: bool = ...,
                 num_epochs: _Optional[int] = ...,
                 learning_rate: _Optional[float] = ...,
                 cpu: _Optional[int] = ...,
                 gpu: _Optional[int] = ...,
                 memory: _Optional[int] = ...,
                 train_test_split: _Optional[float] = ...) -> None: ...


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


class StartEvaluationJobRequest(_message.Message):
    __slots__ = ("type", "base_model_id", "dataset_id", "adapter_id", "cpu", "gpu", "memory")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    BASE_MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_ID_FIELD_NUMBER: _ClassVar[int]
    CPU_FIELD_NUMBER: _ClassVar[int]
    GPU_FIELD_NUMBER: _ClassVar[int]
    MEMORY_FIELD_NUMBER: _ClassVar[int]
    type: EvaluationJobType
    base_model_id: str
    dataset_id: str
    adapter_id: str
    cpu: int
    gpu: int
    memory: int

    def __init__(self,
                 type: _Optional[_Union[EvaluationJobType,
                                        str]] = ...,
                 base_model_id: _Optional[str] = ...,
                 dataset_id: _Optional[str] = ...,
                 adapter_id: _Optional[str] = ...,
                 cpu: _Optional[int] = ...,
                 gpu: _Optional[int] = ...,
                 memory: _Optional[int] = ...) -> None: ...


class StartEvaluationJobResponse(_message.Message):
    __slots__ = ("job",)
    JOB_FIELD_NUMBER: _ClassVar[int]
    job: EvaluationJobMetadata
    def __init__(self, job: _Optional[_Union[EvaluationJobMetadata, _Mapping]] = ...) -> None: ...


class RemoveEvaluationJobRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...


class RemoveEvaluationJobResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...


class GetAppStateRequest(_message.Message):
    __slots__ = ("user",)
    USER_FIELD_NUMBER: _ClassVar[int]
    user: str
    def __init__(self, user: _Optional[str] = ...) -> None: ...


class GetAppStateResponse(_message.Message):
    __slots__ = ("state",)
    STATE_FIELD_NUMBER: _ClassVar[int]
    state: AppState
    def __init__(self, state: _Optional[_Union[AppState, _Mapping]] = ...) -> None: ...


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
    type: DatasetType
    name: str
    description: str
    huggingface_name: str
    location: str
    features: _containers.RepeatedScalarFieldContainer[str]

    def __init__(self,
                 id: _Optional[str] = ...,
                 type: _Optional[_Union[DatasetType,
                                        str]] = ...,
                 name: _Optional[str] = ...,
                 description: _Optional[str] = ...,
                 huggingface_name: _Optional[str] = ...,
                 location: _Optional[str] = ...,
                 features: _Optional[_Iterable[str]] = ...) -> None: ...


class RegisteredModelMetadata(_message.Message):
    __slots__ = ("cml_registered_model_id", "mlflow_experiment_id", "mlflow_run_id")
    CML_REGISTERED_MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MLFLOW_EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    MLFLOW_RUN_ID_FIELD_NUMBER: _ClassVar[int]
    cml_registered_model_id: str
    mlflow_experiment_id: str
    mlflow_run_id: str

    def __init__(
        self,
        cml_registered_model_id: _Optional[str] = ...,
        mlflow_experiment_id: _Optional[str] = ...,
        mlflow_run_id: _Optional[str] = ...) -> None: ...


class ModelMetadata(_message.Message):
    __slots__ = (
        "id",
        "type",
        "framework",
        "name",
        "huggingface_model_name",
        "location",
        "registered_model",
        "bnb_config_id")
    ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    FRAMEWORK_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    HUGGINGFACE_MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    REGISTERED_MODEL_FIELD_NUMBER: _ClassVar[int]
    BNB_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    type: ModelType
    framework: ModelFrameworkType
    name: str
    huggingface_model_name: str
    location: str
    registered_model: RegisteredModelMetadata
    bnb_config_id: str

    def __init__(self,
                 id: _Optional[str] = ...,
                 type: _Optional[_Union[ModelType,
                                        str]] = ...,
                 framework: _Optional[_Union[ModelFrameworkType,
                                             str]] = ...,
                 name: _Optional[str] = ...,
                 huggingface_model_name: _Optional[str] = ...,
                 location: _Optional[str] = ...,
                 registered_model: _Optional[_Union[RegisteredModelMetadata,
                                                    _Mapping]] = ...,
                 bnb_config_id: _Optional[str] = ...) -> None: ...


class AdapterMetadata(_message.Message):
    __slots__ = (
        "id",
        "type",
        "name",
        "model_id",
        "location",
        "huggingface_name",
        "job_id",
        "prompt_id",
        "registered_model",
        "bnb_config_id")
    ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    HUGGINGFACE_NAME_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_ID_FIELD_NUMBER: _ClassVar[int]
    REGISTERED_MODEL_FIELD_NUMBER: _ClassVar[int]
    BNB_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    type: AdapterType
    name: str
    model_id: str
    location: str
    huggingface_name: str
    job_id: str
    prompt_id: str
    registered_model: RegisteredModelMetadata
    bnb_config_id: str

    def __init__(self,
                 id: _Optional[str] = ...,
                 type: _Optional[_Union[AdapterType,
                                        str]] = ...,
                 name: _Optional[str] = ...,
                 model_id: _Optional[str] = ...,
                 location: _Optional[str] = ...,
                 huggingface_name: _Optional[str] = ...,
                 job_id: _Optional[str] = ...,
                 prompt_id: _Optional[str] = ...,
                 registered_model: _Optional[_Union[RegisteredModelMetadata,
                                                    _Mapping]] = ...,
                 bnb_config_id: _Optional[str] = ...) -> None: ...


class PromptMetadata(_message.Message):
    __slots__ = ("id", "name", "dataset_id", "prompt_template")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    dataset_id: str
    prompt_template: str

    def __init__(
        self,
        id: _Optional[str] = ...,
        name: _Optional[str] = ...,
        dataset_id: _Optional[str] = ...,
        prompt_template: _Optional[str] = ...) -> None: ...


class WorkerProps(_message.Message):
    __slots__ = ("num_cpu", "num_memory", "num_gpu")
    NUM_CPU_FIELD_NUMBER: _ClassVar[int]
    NUM_MEMORY_FIELD_NUMBER: _ClassVar[int]
    NUM_GPU_FIELD_NUMBER: _ClassVar[int]
    num_cpu: int
    num_memory: int
    num_gpu: int

    def __init__(
        self,
        num_cpu: _Optional[int] = ...,
        num_memory: _Optional[int] = ...,
        num_gpu: _Optional[int] = ...) -> None: ...


class FineTuningJobMetadata(_message.Message):
    __slots__ = (
        "job_id",
        "base_model_id",
        "dataset_id",
        "prompt_id",
        "num_workers",
        "cml_job_id",
        "adapter_id",
        "worker_props",
        "num_epochs",
        "learning_rate",
        "out_dir",
        "training_arguments_id",
        "model_bnb_config_id",
        "adapter_bnb_config_id")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    BASE_MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_ID_FIELD_NUMBER: _ClassVar[int]
    NUM_WORKERS_FIELD_NUMBER: _ClassVar[int]
    CML_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_PROPS_FIELD_NUMBER: _ClassVar[int]
    NUM_EPOCHS_FIELD_NUMBER: _ClassVar[int]
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    OUT_DIR_FIELD_NUMBER: _ClassVar[int]
    TRAINING_ARGUMENTS_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_BNB_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_BNB_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    base_model_id: str
    dataset_id: str
    prompt_id: str
    num_workers: int
    cml_job_id: str
    adapter_id: str
    worker_props: WorkerProps
    num_epochs: int
    learning_rate: float
    out_dir: str
    training_arguments_id: str
    model_bnb_config_id: str
    adapter_bnb_config_id: str

    def __init__(self,
                 job_id: _Optional[str] = ...,
                 base_model_id: _Optional[str] = ...,
                 dataset_id: _Optional[str] = ...,
                 prompt_id: _Optional[str] = ...,
                 num_workers: _Optional[int] = ...,
                 cml_job_id: _Optional[str] = ...,
                 adapter_id: _Optional[str] = ...,
                 worker_props: _Optional[_Union[WorkerProps,
                                                _Mapping]] = ...,
                 num_epochs: _Optional[int] = ...,
                 learning_rate: _Optional[float] = ...,
                 out_dir: _Optional[str] = ...,
                 training_arguments_id: _Optional[str] = ...,
                 model_bnb_config_id: _Optional[str] = ...,
                 adapter_bnb_config_id: _Optional[str] = ...) -> None: ...


class BnbConfig(_message.Message):
    __slots__ = (
        "load_in_8bit",
        "load_in_4bit",
        "bnb_4bit_compute_dtype",
        "bnb_4bit_quant_type",
        "bnb_4bit_use_double_quant",
        "bnb_4bit_quant_storage",
        "quant_method")
    LOAD_IN_8BIT_FIELD_NUMBER: _ClassVar[int]
    LOAD_IN_4BIT_FIELD_NUMBER: _ClassVar[int]
    BNB_4BIT_COMPUTE_DTYPE_FIELD_NUMBER: _ClassVar[int]
    BNB_4BIT_QUANT_TYPE_FIELD_NUMBER: _ClassVar[int]
    BNB_4BIT_USE_DOUBLE_QUANT_FIELD_NUMBER: _ClassVar[int]
    BNB_4BIT_QUANT_STORAGE_FIELD_NUMBER: _ClassVar[int]
    QUANT_METHOD_FIELD_NUMBER: _ClassVar[int]
    load_in_8bit: bool
    load_in_4bit: bool
    bnb_4bit_compute_dtype: str
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool
    bnb_4bit_quant_storage: str
    quant_method: str

    def __init__(
        self,
        load_in_8bit: bool = ...,
        load_in_4bit: bool = ...,
        bnb_4bit_compute_dtype: _Optional[str] = ...,
        bnb_4bit_quant_type: _Optional[str] = ...,
        bnb_4bit_use_double_quant: bool = ...,
        bnb_4bit_quant_storage: _Optional[str] = ...,
        quant_method: _Optional[str] = ...) -> None: ...


class BnbConfigMetadata(_message.Message):
    __slots__ = ("id", "bnb_config")
    ID_FIELD_NUMBER: _ClassVar[int]
    BNB_CONFIG_FIELD_NUMBER: _ClassVar[int]
    id: str
    bnb_config: BnbConfig
    def __init__(self, id: _Optional[str] = ..., bnb_config: _Optional[_Union[BnbConfig, _Mapping]] = ...) -> None: ...


class TrainingArguments(_message.Message):
    __slots__ = (
        "output_dir",
        "num_train_epochs",
        "optim",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "warmup_ratio",
        "max_grad_norm",
        "learning_rate",
        "fp16",
        "logging_steps",
        "lr_scheduler_type",
        "disable_tqdm",
        "evaluation_strategy",
        "eval_steps",
        "save_strategy",
        "report_to")
    OUTPUT_DIR_FIELD_NUMBER: _ClassVar[int]
    NUM_TRAIN_EPOCHS_FIELD_NUMBER: _ClassVar[int]
    OPTIM_FIELD_NUMBER: _ClassVar[int]
    PER_DEVICE_TRAIN_BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    GRADIENT_ACCUMULATION_STEPS_FIELD_NUMBER: _ClassVar[int]
    WARMUP_RATIO_FIELD_NUMBER: _ClassVar[int]
    MAX_GRAD_NORM_FIELD_NUMBER: _ClassVar[int]
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    FP16_FIELD_NUMBER: _ClassVar[int]
    LOGGING_STEPS_FIELD_NUMBER: _ClassVar[int]
    LR_SCHEDULER_TYPE_FIELD_NUMBER: _ClassVar[int]
    DISABLE_TQDM_FIELD_NUMBER: _ClassVar[int]
    EVALUATION_STRATEGY_FIELD_NUMBER: _ClassVar[int]
    EVAL_STEPS_FIELD_NUMBER: _ClassVar[int]
    SAVE_STRATEGY_FIELD_NUMBER: _ClassVar[int]
    REPORT_TO_FIELD_NUMBER: _ClassVar[int]
    output_dir: str
    num_train_epochs: int
    optim: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    max_grad_norm: float
    learning_rate: float
    fp16: bool
    logging_steps: int
    lr_scheduler_type: str
    disable_tqdm: bool
    evaluation_strategy: str
    eval_steps: int
    save_strategy: str
    report_to: str

    def __init__(
        self,
        output_dir: _Optional[str] = ...,
        num_train_epochs: _Optional[int] = ...,
        optim: _Optional[str] = ...,
        per_device_train_batch_size: _Optional[int] = ...,
        gradient_accumulation_steps: _Optional[int] = ...,
        warmup_ratio: _Optional[float] = ...,
        max_grad_norm: _Optional[float] = ...,
        learning_rate: _Optional[float] = ...,
        fp16: bool = ...,
        logging_steps: _Optional[int] = ...,
        lr_scheduler_type: _Optional[str] = ...,
        disable_tqdm: bool = ...,
        evaluation_strategy: _Optional[str] = ...,
        eval_steps: _Optional[int] = ...,
        save_strategy: _Optional[str] = ...,
        report_to: _Optional[str] = ...) -> None: ...


class TrainingArgumentsMetadata(_message.Message):
    __slots__ = ("id", "training_arguments")
    ID_FIELD_NUMBER: _ClassVar[int]
    TRAINING_ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
    id: str
    training_arguments: TrainingArguments
    def __init__(self, id: _Optional[str] = ...,
                 training_arguments: _Optional[_Union[TrainingArguments, _Mapping]] = ...) -> None: ...


class GenerationConfig(_message.Message):
    __slots__ = ("do_sample", "temperature", "max_new_tokens", "top_p", "top_k", "repetition_penalty", "num_beams")
    DO_SAMPLE_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MAX_NEW_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    REPETITION_PENALTY_FIELD_NUMBER: _ClassVar[int]
    NUM_BEAMS_FIELD_NUMBER: _ClassVar[int]
    do_sample: bool
    temperature: float
    max_new_tokens: int
    top_p: float
    top_k: float
    repetition_penalty: float
    num_beams: int

    def __init__(
        self,
        do_sample: bool = ...,
        temperature: _Optional[float] = ...,
        max_new_tokens: _Optional[int] = ...,
        top_p: _Optional[float] = ...,
        top_k: _Optional[float] = ...,
        repetition_penalty: _Optional[float] = ...,
        num_beams: _Optional[int] = ...) -> None: ...


class GenerationConfigMetadata(_message.Message):
    __slots__ = ("id", "generation_config")
    ID_FIELD_NUMBER: _ClassVar[int]
    GENERATION_CONFIG_FIELD_NUMBER: _ClassVar[int]
    id: str
    generation_config: GenerationConfig
    def __init__(self, id: _Optional[str] = ...,
                 generation_config: _Optional[_Union[GenerationConfig, _Mapping]] = ...) -> None: ...


class EvaluationJobMetadata(_message.Message):
    __slots__ = (
        "job_id",
        "cml_job_id",
        "base_model_id",
        "dataset_id",
        "num_workers",
        "adapter_id",
        "worker_props",
        "evaluation_dir",
        "model_bnb_config_id",
        "adapter_bnb_config_id",
        "generation_arguments_id")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    CML_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    BASE_MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    NUM_WORKERS_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_PROPS_FIELD_NUMBER: _ClassVar[int]
    EVALUATION_DIR_FIELD_NUMBER: _ClassVar[int]
    MODEL_BNB_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_BNB_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    GENERATION_ARGUMENTS_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    cml_job_id: str
    base_model_id: str
    dataset_id: str
    num_workers: int
    adapter_id: str
    worker_props: WorkerProps
    evaluation_dir: str
    model_bnb_config_id: str
    adapter_bnb_config_id: str
    generation_arguments_id: str

    def __init__(self,
                 job_id: _Optional[str] = ...,
                 cml_job_id: _Optional[str] = ...,
                 base_model_id: _Optional[str] = ...,
                 dataset_id: _Optional[str] = ...,
                 num_workers: _Optional[int] = ...,
                 adapter_id: _Optional[str] = ...,
                 worker_props: _Optional[_Union[WorkerProps,
                                                _Mapping]] = ...,
                 evaluation_dir: _Optional[str] = ...,
                 model_bnb_config_id: _Optional[str] = ...,
                 adapter_bnb_config_id: _Optional[str] = ...,
                 generation_arguments_id: _Optional[str] = ...) -> None: ...


class AppState(_message.Message):
    __slots__ = (
        "datasets",
        "models",
        "fine_tuning_jobs",
        "evaluation_jobs",
        "prompts",
        "adapters",
        "training_arguments",
        "bnb_configs",
        "generation_configs")
    DATASETS_FIELD_NUMBER: _ClassVar[int]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    FINE_TUNING_JOBS_FIELD_NUMBER: _ClassVar[int]
    EVALUATION_JOBS_FIELD_NUMBER: _ClassVar[int]
    PROMPTS_FIELD_NUMBER: _ClassVar[int]
    ADAPTERS_FIELD_NUMBER: _ClassVar[int]
    TRAINING_ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
    BNB_CONFIGS_FIELD_NUMBER: _ClassVar[int]
    GENERATION_CONFIGS_FIELD_NUMBER: _ClassVar[int]
    datasets: _containers.RepeatedCompositeFieldContainer[DatasetMetadata]
    models: _containers.RepeatedCompositeFieldContainer[ModelMetadata]
    fine_tuning_jobs: _containers.RepeatedCompositeFieldContainer[FineTuningJobMetadata]
    evaluation_jobs: _containers.RepeatedCompositeFieldContainer[EvaluationJobMetadata]
    prompts: _containers.RepeatedCompositeFieldContainer[PromptMetadata]
    adapters: _containers.RepeatedCompositeFieldContainer[AdapterMetadata]
    training_arguments: _containers.RepeatedCompositeFieldContainer[TrainingArgumentsMetadata]
    bnb_configs: _containers.RepeatedCompositeFieldContainer[BnbConfigMetadata]
    generation_configs: _containers.RepeatedCompositeFieldContainer[GenerationConfigMetadata]

    def __init__(self,
                 datasets: _Optional[_Iterable[_Union[DatasetMetadata,
                                                      _Mapping]]] = ...,
                 models: _Optional[_Iterable[_Union[ModelMetadata,
                                                    _Mapping]]] = ...,
                 fine_tuning_jobs: _Optional[_Iterable[_Union[FineTuningJobMetadata,
                                                              _Mapping]]] = ...,
                 evaluation_jobs: _Optional[_Iterable[_Union[EvaluationJobMetadata,
                                                             _Mapping]]] = ...,
                 prompts: _Optional[_Iterable[_Union[PromptMetadata,
                                                     _Mapping]]] = ...,
                 adapters: _Optional[_Iterable[_Union[AdapterMetadata,
                                                      _Mapping]]] = ...,
                 training_arguments: _Optional[_Iterable[_Union[TrainingArgumentsMetadata,
                                                                _Mapping]]] = ...,
                 bnb_configs: _Optional[_Iterable[_Union[BnbConfigMetadata,
                                                         _Mapping]]] = ...,
                 generation_configs: _Optional[_Iterable[_Union[GenerationConfigMetadata,
                                                                _Mapping]]] = ...) -> None: ...
