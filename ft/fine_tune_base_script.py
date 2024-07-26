
# if st.button("Fine Tune Model"):
#     with st.spinner("Mapping Prompt Template to Dataset"):
#         mapped_dataset = map_dataset_with_prompt_template(dataset, prompt_template)

#     with st.spinner("Preparing Trainer..."):

#         # Load a model using PEFT library for finetuning

from transformers import BitsAndBytesConfig
from pathlib import Path
from peft import LoraConfig
import json
import datasets
from ft import fine_tune
import ft
import argparse
import os

arg_string = os.environ['JOB_ARGUMENTS']

parser = argparse.ArgumentParser()

parser.add_argument(
    "--aggregateconfig",
    help="Path of the aggregate config containing: LoraConfig, bitsandbytesConfig, prompt text, trainerarguments")
parser.add_argument("--loraconfig", help="Path of the LoRA config json")
parser.add_argument("--bnbconfig", help="Path of the BitsandBytes config json")
parser.add_argument("--prompttemplate", help="Path of the PromptTemplate")
parser.add_argument("--trainerarguments", help="Path of the trainer arguments json")
parser.add_argument("--basemodel", help="huggingface basemodel to use")
parser.add_argument("--dataset", help="huggingface dataset to use")
parser.add_argument("--experimentid", help="uuid to use for experiment tracking")
parser.add_argument("--out_dir", help="uuid to use for experiment tracking")


args = parser.parse_args(arg_string.split())
print(args)


DATA_TEXT_FIELD = "prediction"
NUM_EPOCHS = 1.0


# Load aggregate config file
aggregate_config_file = Path(args.aggregateconfig).read_text()
aggregate_config = json.loads(aggregate_config_file)
BitsAndBytesConfig(**aggregate_config['bnb_config'])


finetuner = fine_tune.AMPFineTuner(
    base_model=args.basemodel,
    ft_job_uuid=args.experimentid,
    bnb_config=BitsAndBytesConfig(**aggregate_config['bnb_config'])
)

# Set LoRA training configuration
finetuner.set_lora_config(
    LoraConfig(
        r=16,
        lora_alpha=32,
        # target_modules=["query_key_value", "xxx"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
)

# Set training arguments
# see fine_tuner.py for list of defaults and huggingface's transformers.TrainingArguments
# or the full list of arguments
finetuner.training_args.num_train_epochs = NUM_EPOCHS
finetuner.training_args.warmup_ratio = 0.03
finetuner.training_args.max_grad_norm = 0.3
finetuner.training_args.learning_rate = 2e-4


# Loads a dataset (no need to cache because HF caches by default)
def load_dataset(dataset_name, dataset_fraction=1):
    return datasets.load_dataset(dataset_name, split=f'train[:{dataset_fraction}%]')

# Map a dataset with a new prompt template


def map_dataset_with_prompt_template(dataset: datasets.Dataset, prompt_template: str):
    def ds_map(data):
        data[DATA_TEXT_FIELD] = prompt_template.format(**data)
        return data
    dataset = dataset.map(ds_map)
    return dataset


prompt_text = Path(args.prompttemplate).read_text()
dataset = load_dataset(args.dataset)
mapped_dataset = map_dataset_with_prompt_template(dataset, prompt_text)

finetuner.train(mapped_dataset, DATA_TEXT_FIELD, f"{args.out_dir}")
