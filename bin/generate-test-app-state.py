from uuid import uuid4
import json 
from ft.state import AppState
from ft.api import *
from datasets import load_dataset
from typing import List

OUTPUT_FILE = ".app/state.json"

# TODO: "add dataset" page to add new datasets from HF
AVAILABLE_DATASETS = [
    "Clinton/Text-to-sql-v1",
    "hakurei/open-instruct-v1",
    'philschmid/sql-create-context-copy',
    'teknium/GPTeacher-General-Instruct',
    's-nlp/paradetox',
]


AVAILABLE_MODELS = [
    "distilbert/distilgpt2",
    "Qwen/Qwen2-0.5B",
    "EleutherAI/pythia-160m",
]


TRAINING_PROMPTS = {

    "Clinton/Text-to-sql-v1": 
"""<INSTRUCT>: {instruction}
<INPUT>: {input}
<RESPONSE>: {response}""",

    "hakurei/open-instruct-v1":
"""<INSTRUCTION>: {instruction}
<INPUT>: {input}
<OUTPUT>: {output}"""
}


datasets: List[DatasetMetadata] = [] 
for dataset_name in AVAILABLE_DATASETS:

    # Load the dataset to get the features
    dataset = load_dataset(dataset_name)

    if "train" in dataset:
        dataset = dataset["train"]

    datasets.append(DatasetMetadata(
        id=str(uuid4()),
        name=dataset_name,
        description=dataset.description,
        type=DatasetType.DATASET_TYPE_HUGGINGFACE,
        huggingface_name=dataset_name,
        features=dataset.column_names,
        location=None
    ))


models: List[ModelMetadata] = []
for model_name in AVAILABLE_MODELS:
    models.append(ModelMetadata(
        id=str(uuid4()),
        type=ModelType.MODEL_TYPE_HUGGINGFACE,
        name=model_name,
        huggingface_model_name=model_name
    ))



state: AppState = AppState(
    datasets=datasets,
    models=models,
    jobs=[],
    prompts=[]
)


with open(OUTPUT_FILE, "w") as fout:
    fout.write(state.model_dump_json(indent=2))

    
