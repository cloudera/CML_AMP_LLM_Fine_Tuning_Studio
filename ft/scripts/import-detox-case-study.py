from ft.api import *
from ft.client import FineTuningStudioClient
from typing import Union
from uuid import uuid4

# Import data.

# Detox adapter trained for a mistral 7b with detox dataset.
# Ticketing adapter trained with the "Toxic" ticketing dataset.
# Eval tool used to generate a "detoxified ticketing" dataset, by creating a detoxification prompt specifically for the ticketing dataset.
# "Toxic" ticketing adapter can be evaluated against a "non-toxic" ticketing dataset.

# Imports:
# - detoxifying adapter
# - "toxic ticketing" adapter
# - non-toxic ticketing dataset

# Datasets
import_hf_datasets = ["s-nlp/paradetox", "bitext/Bitext-events-ticketing-llm-chatbot-training-dataset"]
import_csv_datasets = [
    ["Ticketing Actions", "data/datasets/clean_ticketing.csv"]
]


# Models
import_models = ["unsloth/mistral-7b-instruct-v0.3"]

# Adapters
import_adapters = [["mistral-7b-detox",
                    "data/adapters/mistral-7b-detox",
                    "unsloth/mistral-7b-instruct-v0.3",
                    "Detoxifying Prompt (for detox dataset)"],
                   ["mistral-7b-ticketing",
                    "data/adapters/mistral-7b-ticketing",
                    "unsloth/mistral-7b-instruct-v0.3",
                    "Ticketing Prompt with Actions"],
                   ]


# Prompts


input_template = """You are an event ticketing customer LLM chatbot responsible for generating a one-word, snake_case action, based on a customer input. You may only select from one of these actions:

['track_cancellation', 'transfer_ticket', 'upgrade_ticket', 'check_cancellation_policy', 'pay', 'buy_ticket', 'check_cancellation_fee', 'delivery_period', 'get_refund', 'check_refund_policy', 'track_refund', 'cancel_ticket', 'customer_service', 'check_privacy_policy', 'information_about_type_events', 'report_payment_issue', 'find_ticket', 'sell_ticket', 'change_personal_details_on_ticket', 'payment_methods', 'information_about_tickets', 'human_agent', 'delivery_options', 'find_upcoming_events', 'event_organizer']

Please provide the most relevant action based on the input from the customer below.

### CUSTOMER: {instruction}
### ACTION: """
completion_template = """{intent}"""

toxic_input = """You are an LLM responsible for removing toxicity from comments. Given the toxic comment below, please generate an equivalent neutral comment.

<toxic_comment>: {en_toxic_comment}
<neutral_comment>: """
toxic_completion = """{en_neutral_comment}"""

# Prompts
import_prompts = [
    {
        "name": "Ticketing Prompt (clean ticketing dataset)",
        "dataset": "Ticketing Actions",
        "input_template": input_template,
        "completion_template": completion_template
    },
    {
        "name": "Ticketing Prompt (toxic ticketing dataset)",
        "dataset": "bitext/Bitext-events-ticketing-llm-chatbot-training-dataset",
        "input_template": input_template,
        "completion_template": completion_template
    },
    {
        "name": "Detoxifying Prompt (detoxification dataset)",
        "dataset": "s-nlp/paradetox",
        "input_template": toxic_input,
        "completion_template": toxic_completion
    },
    {
        "name": "Detoxifying Prompt (toxic ticketing dataset)",
        "dataset": "bitext/Bitext-events-ticketing-llm-chatbot-training-dataset",
        "input_template": toxic_input,
        "completion_template": toxic_completion
    }
]


# Some helper functions that should be part of our client, or part of our
# Get* request messages so we don't do entire DB calls here.

def get_model_by_hf_name(fts: FineTuningStudioClient, hf_model_name: str) -> Union[ModelMetadata, None]:
    """
    Get a model by the model name.

    todo: extend to other model types
    """
    models = fts.get_models()
    models = list(filter(lambda x: x.huggingface_model_name == hf_model_name, models))
    return models[0] if models else None


def get_prompt_by_name(fts: FineTuningStudioClient, prompt_name: str) -> Union[PromptMetadata, None]:
    prompts = fts.get_prompts()
    prompts = list(filter(lambda x: x.name == prompt_name, prompts))
    return prompts[0] if prompts else None


def get_adapter_by_name(fts: FineTuningStudioClient, adapter_name: str) -> Union[AdapterMetadata, None]:
    adapters = fts.get_adapters()
    adapters = list(filter(lambda x: x.name == adapter_name, adapters))
    return adapters[0] if adapters else None


def get_dataset_by_name(fts: FineTuningStudioClient, dataset_name: str) -> Union[DatasetMetadata, None]:
    datasets = fts.get_datasets()
    datasets = list(filter(lambda x: x.name == dataset_name, datasets))
    return datasets[0] if datasets else None


def force_add_adapter(fts: FineTuningStudioClient, request: AddAdapterRequest) -> AddAdapterResponse:
    """
    If adapter name already exists, return that adapter.
    """
    adapter: AdapterMetadata = get_adapter_by_name(fts, request.name)
    if adapter:
        return AddAdapterResponse(adapter=adapter)
    return fts.AddAdapter(request)


def force_add_dataset(fts: FineTuningStudioClient, request: AddDatasetRequest) -> AddDatasetResponse:
    """
    If dataset name already exists, return that dataset.
    """
    dataset: DatasetMetadata = get_dataset_by_name(fts, request.name)
    if dataset:
        return AddDatasetResponse(dataset=dataset)
    return fts.AddDataset(request)


def force_add_hf_model(fts: FineTuningStudioClient, request: AddModelRequest) -> AddModelResponse:
    """
    If model name already exists, return that model.

    todo: extend to all model types by name
    """
    model: ModelMetadata = get_model_by_hf_name(fts, request.huggingface_name)
    if model:
        return AddModelResponse(model=model)
    return fts.AddModel(request)


def force_add_prompt(fts: FineTuningStudioClient, request: AddPromptRequest) -> AddPromptResponse:
    """
    If prompt name already exists, return that prompt.
    """
    prompt: PromptMetadata = get_prompt_by_name(fts, request.prompt.name)
    if prompt:
        return AddPromptResponse(prompt=prompt)
    return fts.AddPrompt(request)


# Create the client
fts = FineTuningStudioClient()

print("### DATASETS: ")
for hf_ds in import_hf_datasets:
    print(force_add_dataset(fts, AddDatasetRequest(
        type=DatasetType.HUGGINGFACE,
        huggingface_name=hf_ds,
        name=hf_ds
    )).dataset)

for csv_ds in import_csv_datasets:
    print(force_add_dataset(fts, AddDatasetRequest(
        type=DatasetType.PROJECT_CSV,
        name=csv_ds[0],
        location=csv_ds[1]
    )).dataset)

print("### PROMPTS: ")
for prompt in import_prompts:
    # todo: disassemble PromptRequest so id isn't required
    input_prompt = PromptMetadata(
        id=str(uuid4()),
        type=PromptType.IN_PLACE,
        name=prompt["name"],
        dataset_id=get_dataset_by_name(fts, prompt["dataset"]).id,
        input_template=prompt["input_template"],
        completion_template=prompt["completion_template"],
        prompt_template=prompt["input_template"] + prompt["completion_template"]
    )
    print(force_add_prompt(fts, AddPromptRequest(prompt=input_prompt)).prompt)

print("### MODELS: ")
for model in import_models:
    print(force_add_hf_model(fts, AddModelRequest(
        type=ModelType.HUGGINGFACE,
        huggingface_name=model
    )).model)

print("### ADAPTERS: ")
for adapter in import_adapters:
    print(force_add_adapter(fts, AddAdapterRequest(
        type=AdapterType.PROJECT,
        name=adapter[0],
        location=adapter[1],
        model_id=get_model_by_hf_name(fts, adapter[2]).id,  # todo: handle None union
        prompt_id=get_prompt_by_name(fts, adapter[3]).id,
    )).adapter)
