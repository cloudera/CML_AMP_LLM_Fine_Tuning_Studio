from transformers import BitsAndBytesConfig
from pathlib import Path
from peft import LoraConfig
import json
import datasets
from ft import fine_tune
import argparse
import os

# Constants
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
DATA_TEXT_FIELD = "prediction"

# Parse arguments from environment variable
arg_string = os.environ.get('JOB_ARGUMENTS', '')

parser = argparse.ArgumentParser()

parser.add_argument(
    "--aggregateconfig",
    help="Path of the aggregate config containing: LoraConfig, BitsAndBytesConfig, prompt text, trainer arguments",
    required=True
)
parser.add_argument("--loraconfig", help="Path of the LoRA config json")
parser.add_argument("--bnbconfig", help="Path of the BitsAndBytes config json")
parser.add_argument("--prompttemplate", help="Path of the PromptTemplate", required=True)
parser.add_argument("--trainerarguments", help="Path of the trainer arguments json")
parser.add_argument("--basemodel", help="Huggingface base model to use", required=True)
parser.add_argument("--dataset", help="Huggingface dataset to use", required=True)
parser.add_argument("--experimentid", help="UUID to use for experiment tracking", required=True)
parser.add_argument("--out_dir", help="Output directory for the fine-tuned model", required=True)
parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Epochs for fine tuning job")
parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate for fine tuning job")

args = parser.parse_args(arg_string.split())

# Load aggregate config file
try:
    aggregate_config_file = Path(args.aggregateconfig).read_text()
    aggregate_config = json.loads(aggregate_config_file)
except (FileNotFoundError, json.JSONDecodeError) as e:
    raise RuntimeError(f"Error loading aggregate config file: {e}")

# Initialize BitsAndBytesConfig
try:
    bnb_config = BitsAndBytesConfig(**aggregate_config['bnb_config'])
except KeyError as e:
    raise KeyError(f"Missing key in BitsAndBytesConfig: {e}")

# Initialize the fine-tuner
finetuner = fine_tune.AMPFineTuner(
    base_model=args.basemodel,
    ft_job_uuid=args.experimentid,
    bnb_config=bnb_config
)

# Set LoRA training configuration
finetuner.set_lora_config(
    LoraConfig(
        r=16,
        lora_alpha=32,
        # target_modules=["query_key_value", "xxx"],  # Customize as needed
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
)

# Set training arguments
finetuner.training_args.num_train_epochs = args.num_epochs
finetuner.training_args.warmup_ratio = 0.03
finetuner.training_args.max_grad_norm = 0.3
finetuner.training_args.learning_rate = args.learning_rate


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
            data[DATA_TEXT_FIELD] = prompt_template.format(**data)
        except KeyError as e:
            raise KeyError(f"Error formatting data with prompt template: {e}")
        return data

    return dataset.map(ds_map)


# Load and map dataset
try:
    prompt_text = Path(args.prompttemplate).read_text()
    dataset = load_dataset(args.dataset)
    mapped_dataset = map_dataset_with_prompt_template(dataset, prompt_text)
except FileNotFoundError as e:
    raise RuntimeError(f"Error loading prompt template: {e}")

# Start fine-tuning
finetuner.train(mapped_dataset, DATA_TEXT_FIELD, args.out_dir)
