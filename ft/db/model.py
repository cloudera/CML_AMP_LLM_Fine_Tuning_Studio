from sqlalchemy import Column, Integer, String, Text, ForeignKey, Double
from sqlalchemy.orm import declarative_base
from google.protobuf.message import Message
from sqlalchemy.inspection import inspect


# Define the declarative base
Base = declarative_base()


class MappedDict:

    @classmethod
    def from_dict(cls, d: dict):
        """
        Create a model item from a dictionary
        """
        return cls(**d)

    def to_dict(self):
        """
        Extract all of the set key values from an ORM response
        and return a dictionary of key-value pairs.
        """
        result = {}
        for column in inspect(self).mapper.column_attrs:
            value = getattr(self, column.key)
            if value is not None:  # Only include set (non-null) fields
                result[column.key] = value
        return result


class MappedProtobuf:
    """
    Ineriting a MappedProtobuf inherits a class method that
    generates a specific ORM model class from a given protobuf message.
    This method extracts all set fields in a message, name and value, and
    directly passes these as kwargs to a new base model class. Given that the
    majority of our protobuf messages map directly to the name of a column
    in our databases, this can be used to avoid a ton of transformation logic.
    Protobuf > 3.15 allows for .ListFields() to determine which fields have
    been set.
    """

    @classmethod
    def from_message(cls, message: Message):
        """
        Generate this ORM base model from a protobuf message.
        """

        # Determine the fields that are set in the message.
        set_fields = message.ListFields()
        class_kwargs = {}
        for field, value in set_fields:
            if hasattr(cls, field.name):
                class_kwargs[field.name] = value

        return cls(**class_kwargs)

    def to_protobuf(self, protobuf_cls):
        """
        Convert an ORM model to a protobuf message. Any fields
        that directly match in the protobuf message will be mapped
        if the field exists and is non-null in the base model class.
        """
        obj_dict = self.to_dict()
        protobuf_message = protobuf_cls()

        for key, value in obj_dict.items():
            if isinstance(value, list):
                raise ValueError("This method doesn't support setting repeated string types.")
            if hasattr(protobuf_message, key):
                setattr(protobuf_message, key, value)

        return protobuf_message


class Model(Base, MappedProtobuf, MappedDict):
    __tablename__ = 'models'
    id = Column(String, primary_key=True, nullable=False)
    type = Column(String, nullable=True)
    framework = Column(String, nullable=True)
    name = Column(String, nullable=True)
    description = Column(String, nullable=True)
    huggingface_model_name = Column(String, nullable=True)
    location = Column(String, nullable=True)
    cml_registered_model_id = Column(String, nullable=True)
    mlflow_experiment_id = Column(String, nullable=True)
    mlflow_run_id = Column(String, nullable=True)


class Dataset(Base, MappedProtobuf, MappedDict):
    __tablename__ = "datasets"
    id = Column(String, primary_key=True, nullable=False)
    type = Column(String, nullable=True)
    name = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    huggingface_name = Column(String, nullable=True)
    location = Column(Text, nullable=True)
    features = Column(Text, nullable=True)  # Store JSON as TEXT


class Adapter(Base, MappedProtobuf, MappedDict):
    __tablename__ = "adapters"
    id = Column(String, primary_key=True, nullable=False)
    type = Column(String, nullable=True)
    name = Column(String, nullable=True)
    description = Column(String, nullable=True)
    huggingface_name = Column(String, nullable=True)
    model_id = Column(String, ForeignKey('models.id'), nullable=True)
    location = Column(Text, nullable=True)
    fine_tuning_job_id = Column(String, ForeignKey('fine_tuning_jobs.id'), nullable=True)
    prompt_id = Column(String, ForeignKey('prompts.id'), nullable=True)
    cml_registered_model_id = Column(String, nullable=True)
    mlflow_experiment_id = Column(String, nullable=True)
    mlflow_run_id = Column(String, nullable=True)


class Prompt(Base, MappedProtobuf, MappedDict):
    __tablename__ = "prompts"
    id = Column(String, primary_key=True, nullable=False)
    type = Column(String, nullable=True)
    name = Column(String, nullable=True)
    description = Column(String, nullable=True)
    dataset_id = Column(String, ForeignKey('datasets.id'), nullable=True)
    prompt_template = Column(String)
    input_template = Column(String)
    completion_template = Column(String)


class FineTuningJob(Base, MappedProtobuf, MappedDict):
    __tablename__ = "fine_tuning_jobs"
    id = Column(String, primary_key=True, nullable=False)
    base_model_id = Column(String, ForeignKey('models.id'), nullable=True)
    dataset_id = Column(String, ForeignKey('datasets.id'), nullable=True)
    prompt_id = Column(String, ForeignKey('prompts.id'), nullable=True)
    num_workers = Column(Integer, nullable=True)
    cml_job_id = Column(String, nullable=True)
    adapter_id = Column(String, ForeignKey('adapters.id'), nullable=True)
    num_cpu = Column(Integer, nullable=True)
    num_gpu = Column(Integer, nullable=True)
    num_memory = Column(Integer, nullable=True)
    num_epochs = Column(Integer, nullable=True)
    learning_rate = Column(Double, nullable=True)
    out_dir = Column(String, nullable=True)
    training_arguments_config_id = Column(String, ForeignKey('configs.id'), nullable=True)
    model_bnb_config_id = Column(String, ForeignKey('configs.id'), nullable=True)
    adapter_bnb_config_id = Column(String, ForeignKey('configs.id'), nullable=True)
    lora_config_id = Column(String, ForeignKey('configs.id'), nullable=True)
    training_arguments_config = Column(String, nullable=True)
    model_bnb_config = Column(String, nullable=True)
    adapter_bnb_config = Column(String, nullable=True)
    lora_config = Column(String, nullable=True)
    dataset_fraction = Column(Double, nullable=True)
    train_test_split = Column(Double, nullable=True)
    user_script = Column(String, nullable=True)
    user_config_id = Column(String, ForeignKey('configs.id'), nullable=True)
    framework_type = Column(String, nullable=True)
    axolotl_config_id = Column(String, ForeignKey('configs.id'), nullable=True)
    gpu_label_id = Column(Integer, nullable=True)
    adapter_name = Column(String, nullable=True)


class EvaluationJob(Base, MappedProtobuf, MappedDict):
    __tablename__ = "evaluation_jobs"
    id = Column(String, primary_key=True, nullable=False)
    type = Column(String, nullable=True)
    cml_job_id = Column(String, nullable=True)
    parent_job_id = Column(String, nullable=True)
    base_model_id = Column(String, ForeignKey('models.id'), nullable=True)
    dataset_id = Column(String, ForeignKey('datasets.id'), nullable=True)
    prompt_id = Column(String, ForeignKey('prompts.id'), nullable=True)
    num_workers = Column(Integer, nullable=True)
    adapter_id = Column(String, ForeignKey('adapters.id'), nullable=True)
    num_cpu = Column(Integer, nullable=True)
    num_gpu = Column(Integer, nullable=True)
    num_memory = Column(Integer, nullable=True)
    evaluation_dir = Column(String, nullable=True)
    model_bnb_config_id = Column(String, ForeignKey('configs.id'), nullable=True)
    adapter_bnb_config_id = Column(String, ForeignKey('configs.id'), nullable=True)
    generation_config_id = Column(String, ForeignKey('configs.id'), nullable=True)
    model_bnb_config = Column(String, nullable=True)
    adapter_bnb_config = Column(String, nullable=True)
    generation_config = Column(String, nullable=True)


class Config(Base, MappedProtobuf, MappedDict):
    __tablename__ = "configs"
    id = Column(String, primary_key=True, nullable=False)
    type = Column(String, nullable=True)
    description = Column(String, nullable=True)
    config = Column(Text, nullable=True)  # Store JSON as TEXT
    model_family = Column(String, nullable=True)
    is_default = Column(Integer, nullable=True, default=1)


TABLE_TO_MODEL_REGISTRY = {
    'datasets': Dataset,
    'models': Model,
    'prompts': Prompt,
    'adapters': Adapter,
    'fine_tuning_jobs': FineTuningJob,
    'evaluation_jobs': EvaluationJob,
    'configs': Config
}
"""
Mapping of table names to declarative base models. If adding a new schema table,
the new table needs to be appended to this list in order for database import/export
features to work properly.

Ideally we can try to automate this with Base.registry._class_registry.values() or
some other mechanism in the future, but because we move from PascalCase to snake_case,
there will be some processing mechanisms involved.
"""


MODEL_TO_TABLE_REGISTRY = {v: k for k, v in TABLE_TO_MODEL_REGISTRY.items()}
"""
Mapping of declarative base model classes to their table names.

Ideally we can try to automate this with Base.registry._class_registry.values() or
some other mechanism in the future, but because we move from PascalCase to snake_case,
there will be some processing mechanisms involved.
"""
