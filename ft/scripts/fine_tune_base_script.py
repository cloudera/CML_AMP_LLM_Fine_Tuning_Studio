from transformers import BitsAndBytesConfig, TrainingArguments
from pathlib import Path
from peft import LoraConfig
import json
import datasets
from ft import fine_tune
import argparse
import os
from typing import Tuple
from ft.utils import attempt_hf_login
from ft.client import FineTuningStudioClient
from ft.api import *

# Constants
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
DATA_TEXT_FIELD = "prediction"
TRAIN_TEST_SPLIT = 0.1
SEED = 42

# Parse arguments from environment variable
arg_string = os.environ.get('JOB_ARGUMENTS', '')

parser = argparse.ArgumentParser()

parser.add_argument("--prompttemplate", help="Path of the PromptTemplate", required=True)
parser.add_argument("--trainerarguments", help="Path of the trainer arguments json")
parser.add_argument("--basemodel", help="Huggingface base model to use", required=True)
parser.add_argument("--dataset_id", help="Dataset ID from the Fine Tuning Studio application", required=True)
parser.add_argument("--experimentid", help="UUID to use for experiment tracking", required=True)
parser.add_argument("--out_dir", help="Output directory for the fine-tuned model", required=True)
parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Epochs for fine tuning job")
parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate for fine tuning job")
parser.add_argument("--train_test_split", type=float, default=TRAIN_TEST_SPLIT,
                    help="Split of the existing dataset between training and testing.")
parser.add_argument("--bnb_config_id", default=None, help="ID of the BnB config in FT Studio's config store.")
parser.add_argument("--lora_config_id", default=None, help="ID of the Lora config in FT Studio's config store.")
parser.add_argument("--training_arguments_config_id", default=None,
                    help="ID of the training arguments in FT Studio's config store.")
parser.add_argument("--hf_token", help="Huggingface access token to use for gated models", default=None)
parser.add_argument("--fts_server_ip", help="IP address of the FTS gRPC server.", required=True)
parser.add_argument("--fts_server_port", help="Exposed port of the gRPC server", required=True)

args = parser.parse_args(arg_string.split())

# Create a client connection to the FTS server
fts: FineTuningStudioClient = FineTuningStudioClient(server_ip=args.fts_server_ip, server_port=args.fts_server_port)

# Attempt log in to huggingface
attempt_hf_login(args.hf_token)

# Get the configurations.
lora_config_dict = json.loads(
    fts.GetConfig(
        GetConfigRequest(
            id=args.lora_config_id
        )
    ).config
)

bnb_config_dict = json.loads(
    fts.GetConfig(
        GetConfigRequest(
            id=args.bnb_config_id
        )
    ).config
)

training_args_dict = json.loads(
    fts.GetConfig(
        GetConfigRequest(
            id=args.training_arguments_config_id
        )
    ).config
)

# Initialize the fine-tuner
finetuner = fine_tune.AMPFineTuner(
    base_model=args.basemodel,
    ft_job_uuid=args.experimentid,
    bnb_config=BitsAndBytesConfig(**bnb_config_dict),
    training_args=TrainingArguments(**training_args_dict),
    auth_token=args.hf_token,
)

# Set LoRA training configuration
finetuner.set_lora_config(LoraConfig(**lora_config_dict))


def load_dataset(dataset_name, dataset_fraction=100):
    """
    Loads a dataset from Huggingface, optionally sampling a fraction of it.

    Parameters:
        dataset_name (str): The name of the Huggingface dataset to load.
        dataset_fraction (int): The percentage of the dataset to load. Defaults to 100.

    Returns:
        datasets.Dataset: The loaded dataset.
    """
    try:
        return datasets.load_dataset(dataset_name, split=f'train[:{dataset_fraction}%]')
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {e}")


def split_dataset(ds: datasets.Dataset, split_fraction: float = TRAIN_TEST_SPLIT,
                  seed: int = SEED) -> Tuple[datasets.Dataset, datasets.Dataset]:
    """
    Split a dataset into two datasets given a split size and a random seed. This is
    primarily used to create a train dataset and an evaluation dataset.

    Parameters:
        split_fraction (float): the dataset split. The first dataset returned will be of size S*(1-split_fraction).
        seed (int): randomized seed for dataset splitting.

    Returns:
        Tuple[Dataset, Dataset], the two split datasets.
    """
    dataset_split = ds.train_test_split(test_size=split_fraction, shuffle=True, seed=SEED)
    return dataset_split['train'], dataset_split['test']


def map_dataset_with_prompt_template(dataset, prompt_template):
    """
    Maps a dataset with a given prompt template.

    Parameters:
        dataset (datasets.Dataset): The dataset to map.
        prompt_template (str): The prompt template to apply to the dataset.

    Returns:
        datasets.Dataset: The mapped dataset.
    """
    def ds_map(data):
        try:
            data[DATA_TEXT_FIELD] = prompt_template.format(**data) + finetuner.tokenizer.eos_token
        except KeyError as e:
            raise KeyError(f"Error formatting data with prompt template: {e}")
        return data

    return dataset.map(ds_map)


# Load and map dataset
try:
    prompt_text = Path(args.prompttemplate).read_text()

    # Load the base dataset into memory. Call the FTS server
    # to extract metadata information about the dataset. Right now,
    # only huggingface datasets are supported for fine tuning jobs.
    dataset_id = args.dataset_id
    dataset_metadata: DatasetMetadata = fts.GetDataset(
        GetDatasetRequest(
            id=dataset_id
        )
    )
    assert dataset_metadata.type == DatasetType.DATASET_TYPE_HUGGINGFACE
    dataset = load_dataset(dataset_metadata.huggingface_name)

    # Split the above dataset into a training dataset and a testing dataset.
    ds_train, ds_eval = split_dataset(dataset, args.train_test_split)

    # Map both datasets with prompt templates
    ds_train = map_dataset_with_prompt_template(ds_train, prompt_text)
    ds_eval = map_dataset_with_prompt_template(ds_eval, prompt_text)

except FileNotFoundError as e:
    raise RuntimeError(f"Error loading prompt template: {e}")

# Start fine-tuning
finetuner.train(
    train_dataset=ds_train,
    eval_dataset=ds_eval,
    dataset_text_field=DATA_TEXT_FIELD,
    output_dir=args.out_dir)
