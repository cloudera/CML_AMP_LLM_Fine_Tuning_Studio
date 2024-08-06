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


class ImportDatasetRequest(_message.Message):
    __slots__ = ("type", "huggingface_name", "location")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    HUGGINGFACE_NAME_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    type: DatasetType
    huggingface_name: str
    location: str
    def __init__(self, type: _Optional[_Union[DatasetType, str]] = ...,
                 huggingface_name: _Optional[str] = ..., location: _Optional[str] = ...) -> None: ...


class ImportDatasetResponse(_message.Message):
    __slots__ = ("dataset",)
    DATASET_FIELD_NUMBER: _ClassVar[int]
    dataset: DatasetMetadata
    def __init__(self, dataset: _Optional[_Union[DatasetMetadata, _Mapping]] = ...) -> None: ...


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
    __slots__ = ("id", "type", "framework", "name", "huggingface_model_name", "location", "registered_model")
    ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    FRAMEWORK_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    HUGGINGFACE_MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    REGISTERED_MODEL_FIELD_NUMBER: _ClassVar[int]
    id: str
    type: ModelType
    framework: ModelFrameworkType
    name: str
    huggingface_model_name: str
    location: str
    registered_model: RegisteredModelMetadata

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
                                                    _Mapping]] = ...) -> None: ...


class ImportModelRequest(_message.Message):
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


class ImportModelResponse(_message.Message):
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
        "registered_model")
    ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    HUGGINGFACE_NAME_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_ID_FIELD_NUMBER: _ClassVar[int]
    REGISTERED_MODEL_FIELD_NUMBER: _ClassVar[int]
    id: str
    type: AdapterType
    name: str
    model_id: str
    location: str
    huggingface_name: str
    job_id: str
    prompt_id: str
    registered_model: RegisteredModelMetadata

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
                                                    _Mapping]] = ...) -> None: ...


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
        "learning_rate")
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
    job_id: str
    base_model_id: str
    dataset_id: str
    prompt_id: str
    num_workers: str
    cml_job_id: str
    adapter_id: str
    worker_props: WorkerProps
    num_epochs: int
    learning_rate: float

    def __init__(self,
                 job_id: _Optional[str] = ...,
                 base_model_id: _Optional[str] = ...,
                 dataset_id: _Optional[str] = ...,
                 prompt_id: _Optional[str] = ...,
                 num_workers: _Optional[str] = ...,
                 cml_job_id: _Optional[str] = ...,
                 adapter_id: _Optional[str] = ...,
                 worker_props: _Optional[_Union[WorkerProps,
                                                _Mapping]] = ...,
                 num_epochs: _Optional[int] = ...,
                 learning_rate: _Optional[float] = ...) -> None: ...


class BnbConfig(_message.Message):
    __slots__ = (
        "load_in_8bit",
        "load_in_4bit",
        "bnb_4bit_compute_type",
        "bnb_4bit_quant_type",
        "bnb_4bit_use_double_quant",
        "bnb_4bit_quant_storage")
    LOAD_IN_8BIT_FIELD_NUMBER: _ClassVar[int]
    LOAD_IN_4BIT_FIELD_NUMBER: _ClassVar[int]
    BNB_4BIT_COMPUTE_TYPE_FIELD_NUMBER: _ClassVar[int]
    BNB_4BIT_QUANT_TYPE_FIELD_NUMBER: _ClassVar[int]
    BNB_4BIT_USE_DOUBLE_QUANT_FIELD_NUMBER: _ClassVar[int]
    BNB_4BIT_QUANT_STORAGE_FIELD_NUMBER: _ClassVar[int]
    load_in_8bit: bool
    load_in_4bit: bool
    bnb_4bit_compute_type: str
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool
    bnb_4bit_quant_storage: str

    def __init__(
        self,
        load_in_8bit: bool = ...,
        load_in_4bit: bool = ...,
        bnb_4bit_compute_type: _Optional[str] = ...,
        bnb_4bit_quant_type: _Optional[str] = ...,
        bnb_4bit_use_double_quant: bool = ...,
        bnb_4bit_quant_storage: _Optional[str] = ...) -> None: ...


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
    __slots__ = ("job",)
    JOB_FIELD_NUMBER: _ClassVar[int]
    job: FineTuningJobMetadata
    def __init__(self, job: _Optional[_Union[FineTuningJobMetadata, _Mapping]] = ...) -> None: ...


class MLflowEvaluationJobMetadata(_message.Message):
    __slots__ = ("job_id", "cml_job_id", "base_model_id", "dataset_id", "num_workers", "adapter_id", "worker_props")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    CML_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    BASE_MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    NUM_WORKERS_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_PROPS_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    cml_job_id: str
    base_model_id: str
    dataset_id: str
    num_workers: int
    adapter_id: str
    worker_props: WorkerProps

    def __init__(self,
                 job_id: _Optional[str] = ...,
                 cml_job_id: _Optional[str] = ...,
                 base_model_id: _Optional[str] = ...,
                 dataset_id: _Optional[str] = ...,
                 num_workers: _Optional[int] = ...,
                 adapter_id: _Optional[str] = ...,
                 worker_props: _Optional[_Union[WorkerProps,
                                                _Mapping]] = ...) -> None: ...


class StartMLflowEvaluationJobRequest(_message.Message):
    __slots__ = ("base_model_id", "dataset_id", "adapter_id", "cpu", "gpu", "memory")
    BASE_MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_ID_FIELD_NUMBER: _ClassVar[int]
    CPU_FIELD_NUMBER: _ClassVar[int]
    GPU_FIELD_NUMBER: _ClassVar[int]
    MEMORY_FIELD_NUMBER: _ClassVar[int]
    base_model_id: str
    dataset_id: str
    adapter_id: str
    cpu: int
    gpu: int
    memory: int

    def __init__(
        self,
        base_model_id: _Optional[str] = ...,
        dataset_id: _Optional[str] = ...,
        adapter_id: _Optional[str] = ...,
        cpu: _Optional[int] = ...,
        gpu: _Optional[int] = ...,
        memory: _Optional[int] = ...) -> None: ...


class StartMLflowEvaluationJobResponse(_message.Message):
    __slots__ = ("job",)
    JOB_FIELD_NUMBER: _ClassVar[int]
    job: MLflowEvaluationJobMetadata
    def __init__(self, job: _Optional[_Union[MLflowEvaluationJobMetadata, _Mapping]] = ...) -> None: ...


class AppState(_message.Message):
    __slots__ = ("datasets", "models", "jobs", "mlflow", "prompts", "adapters")
    DATASETS_FIELD_NUMBER: _ClassVar[int]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    JOBS_FIELD_NUMBER: _ClassVar[int]
    MLFLOW_FIELD_NUMBER: _ClassVar[int]
    PROMPTS_FIELD_NUMBER: _ClassVar[int]
    ADAPTERS_FIELD_NUMBER: _ClassVar[int]
    datasets: _containers.RepeatedCompositeFieldContainer[DatasetMetadata]
    models: _containers.RepeatedCompositeFieldContainer[ModelMetadata]
    jobs: _containers.RepeatedCompositeFieldContainer[FineTuningJobMetadata]
    mlflow: _containers.RepeatedCompositeFieldContainer[MLflowEvaluationJobMetadata]
    prompts: _containers.RepeatedCompositeFieldContainer[PromptMetadata]
    adapters: _containers.RepeatedCompositeFieldContainer[AdapterMetadata]

    def __init__(self,
                 datasets: _Optional[_Iterable[_Union[DatasetMetadata,
                                                      _Mapping]]] = ...,
                 models: _Optional[_Iterable[_Union[ModelMetadata,
                                                    _Mapping]]] = ...,
                 jobs: _Optional[_Iterable[_Union[FineTuningJobMetadata,
                                                  _Mapping]]] = ...,
                 mlflow: _Optional[_Iterable[_Union[MLflowEvaluationJobMetadata,
                                                    _Mapping]]] = ...,
                 prompts: _Optional[_Iterable[_Union[PromptMetadata,
                                                     _Mapping]]] = ...,
                 adapters: _Optional[_Iterable[_Union[AdapterMetadata,
                                                      _Mapping]]] = ...) -> None: ...
