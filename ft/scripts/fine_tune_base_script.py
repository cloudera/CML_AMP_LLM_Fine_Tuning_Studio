from transformers import BitsAndBytesConfig, TrainingArguments
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
DATA_TEXT_FIELD = "prediction"
TRAIN_TEST_SPLIT = 0.9
DATASET_FRACTION = 1.0
SEED = 42

# Parse arguments from environment variable
arg_string = os.environ.get('JOB_ARGUMENTS', '')

parser = argparse.ArgumentParser()

parser.add_argument("--prompt_id", help="ID of the prompt template to use.", required=True)
parser.add_argument("--base_model_id", help="Base model ID to use.", required=True)
parser.add_argument("--dataset_id", help="Dataset ID from the Fine Tuning Studio application", required=True)
parser.add_argument("--experimentid", help="UUID to use for experiment tracking", required=True)
parser.add_argument("--out_dir", help="Output directory for the fine-tuned model", required=True)
parser.add_argument("--train_out_dir", help="Output directory for the training runs", required=True)
parser.add_argument("--train_test_split", type=float, default=TRAIN_TEST_SPLIT,
                    help="Split of the existing dataset between training and testing.")
parser.add_argument("--dataset_fraction", type=float, default=DATASET_FRACTION,
                    help="Fraction of the dataset to downsample to.")
parser.add_argument("--bnb_config_id", default=None, help="ID of the BnB config in FT Studio's config store.")
parser.add_argument("--lora_config_id", default=None, help="ID of the Lora config in FT Studio's config store.")
parser.add_argument("--training_arguments_config_id", default=None,
                    help="ID of the training arguments in FT Studio's config store.")
parser.add_argument("--hf_token", help="Huggingface access token to use for gated models", default=None)
parser.add_argument("--adapter_name", help="Human friendly name of the adapter to train", default=None)
parser.add_argument(
    "--auto_add_adapter",
    action="store_true",
    help="Automatically add an adapter to database if training succeeds.")

args = parser.parse_args(arg_string.split())

# Create a client connection to the FTS server
fts: FineTuningStudioClient = FineTuningStudioClient()

# Attempt log in to huggingface
attempt_hf_login(args.hf_token)

# Get the configurations.
lora_config_dict = json.loads(
    fts.GetConfig(
        GetConfigRequest(
            id=args.lora_config_id
        )
    ).config.config
)

bnb_config_dict = json.loads(
    fts.GetConfig(
        GetConfigRequest(
            id=args.bnb_config_id
        )
    ).config.config
)

training_args_dict = json.loads(
    fts.GetConfig(
        GetConfigRequest(
            id=args.training_arguments_config_id
        )
    ).config.config
)

# Load the base dataset into memory. Call the FTS server
# to extract metadata information about the dataset. Right now,
# only huggingface datasets are supported for fine tuning jobs.
dataset_id = args.dataset_id
dataset_metadata: DatasetMetadata = fts.GetDataset(
    GetDatasetRequest(
        id=dataset_id
    )
).dataset
assert dataset_metadata.type == DatasetType.HUGGINGFACE

# Extract other fields like base model and prompt.
base_model_md: ModelMetadata = fts.GetModel(
    GetModelRequest(
        id=args.base_model_id
    )
).model

# Extract prompt template information.
prompt_md: PromptMetadata = fts.GetPrompt(
    GetPromptRequest(
        id=args.prompt_id
    )
).prompt

# Override the training args based on the provided output dir. The reason
# this happens within the job (rather than passing the training job dir as part
# of the output config) is that we set the training config BEFORE we have this
# desired job ID field available. This is a side effect of using the UI.
training_args_dict["output_dir"] = args.train_out_dir

# Initialize the fine-tuner
finetuner = fine_tune.AMPFineTuner(
    base_model=base_model_md.huggingface_model_name,
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
        split_fraction (float): the dataset split. The first dataset returned will be of size S*split_fraction.
        seed (int): randomized seed for dataset splitting.

    Returns:
        Tuple[Dataset, Dataset], the two split datasets.
    """
    dataset_split = ds.train_test_split(test_size=(1.0 - split_fraction), shuffle=True, seed=SEED)
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
    prompt_text = prompt_md.prompt_template

    dataset = load_dataset(dataset_metadata.huggingface_name, dataset_fraction=int(100 * args.dataset_fraction))

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


# Upon success, add the adapter to metadata if it's
# requested to do so.
if args.auto_add_adapter:
    fts.AddAdapter(
        AddAdapterRequest(
            type=AdapterType.PROJECT,
            name=args.adapter_name,
            model_id=base_model_md.id,
            location=args.out_dir,
            fine_tuning_job_id=args.experimentid,
            prompt_id=prompt_md.id
        )
    )
