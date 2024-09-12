from torch.nn import Module

from ft.consts import (
    TRAINING_DEFAULT_DATASET_FRACTION,
    TRAINING_DEFAULT_TRAIN_TEST_SPLIT,
    TRAINING_DATA_TEXT_FIELD,
    TRAINING_DATASET_SEED
)

import datasets

from transformers import PreTrainedTokenizerBase

from typing import Tuple, Union


def get_model_parameters(model: Module) -> Tuple[int, int]:
    """Get the total number of parameters, and total number
    of trained parameters, from a PyTorch module.

    Args:
        model (Module): the model in question

    Returns:
        Tuple[int, int]: the total parameters [0], and total trainable parameters [1]
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return all_param, trainable_params


def map_dataset_with_prompt_template(
    dataset: datasets.Dataset,
    prompt_template,
    data_text_field: str = TRAINING_DATA_TEXT_FIELD,
    add_eos_token: bool = True,
    eos_token: str = None
):
    """
    Maps a dataset with a given prompt template.

    Parameters:
        dataset (datasets.Dataset): The dataset to map.
        prompt_template (str): The prompt template to apply to the dataset.

    Returns:
        datasets.Dataset: The mapped dataset.
    """
    def ds_map(data):
        data[data_text_field] = prompt_template.format(**data)
        if add_eos_token and eos_token is not None:
            data[data_text_field] = prompt_template.format(**data) + eos_token
        return data

    try:
        print("before dataset: ", dataset)
        dataset = dataset.map(ds_map)
        print("after dataset: ", dataset)
        return dataset
    except KeyError as e:
        raise KeyError(f"Error formatting data with prompt template: {e}")


def find_padding_token_candidate(tokenizer: PreTrainedTokenizerBase) -> Union[str, None]:
    """
    Find a candidate token that can be used as a padding token which
    already exists in a tokenizer dictionary. This is a very basic heuristic
    and should be expanded in the future to handle more models that have
    undefined padding tokens, yet available special reserved tokens.
    """

    for token in list(tokenizer.added_tokens_encoder.keys()):
        if "pad" in token:
            return token

    return None


def configure_tokenizer_padding(tokenizer: PreTrainedTokenizerBase, pad_token: str = None) -> PreTrainedTokenizerBase:
    """
    Configure a tokenizer's padding token to be a dedicated padding token. If requested, a padding token
    can be set manually based on a model's available tokens.
    """

    # If the tokenizer is already configured with a padding token, and that
    # padding token is dedicated separate from the EOS token, then the
    # tokenizer is configured for TRL training.
    if (
        tokenizer.pad_token_id != -1 and
        tokenizer.pad_token is not None and
        tokenizer.pad_token != tokenizer.eos_token
    ):
        return tokenizer

    # If the tokenzer's padding token is not unique, or if the padding token is
    # not set, and if there's an available token that we can use for padding, then
    # use that token.
    if (
        (
            tokenizer.pad_token is None or
            tokenizer.pad_token_id == -1 or
            tokenizer.pad_token == tokenizer.eos_token
        ) and
        pad_token is not None and pad_token != ""
    ):
        tokenizer_len = len(tokenizer)
        tokenizer.add_special_tokens({'pad_token': pad_token})

        # Right now, if creating a new custom token, this involves changing the model head
        # layer corresponding to embeddings. Because the Studio doesn't allow for changing
        # of the base model layers (yet), there's no way to set a manual token. This token must
        # be set to something that's already available in the tokenizer's vocabulary.
        if len(tokenizer) != tokenizer_len:
            raise ValueError(
                "Right now, a custom padding token needs to be set to a token that's already in the tokenizer's vocabulary.")

        return tokenizer

    # If there is a unique UNK token available, default to using this token.
    if tokenizer.unk_token is not None and tokenizer.unk_token != tokenizer.eos_token:
        tokenizer.add_special_tokens({'pad_token', tokenizer.unk_token})
        return tokenizer

    # If there are any available reserved special tokens that may suit the needs of
    # this specific padding token, then set this special token. This is currently based
    # on a simple heuristic and will help with models that have padding tokens as "reserved"
    # tokens but are not yet set as the actual pad token.
    #
    # Examples of this:
    # NousResearch/Meta-Llama-3.1-8B    -> "<|finetune_right_pad_id|>"
    # EleutherAI/pythia-160m            -> "<|padding|>"
    #
    # Given that we want to enable training for as many models as possible, this heuristic
    # allows for more model matches for our customers.
    padding_token_candidate = find_padding_token_candidate(tokenizer)
    if padding_token_candidate is not None:
        tokenizer.add_special_tokens({'pad_token': padding_token_candidate})
        return tokenizer

    # If we made it here, the tokenizer doesn't have a padding token configuration that
    # is suitable for TRL training. Adapter performance will be unreliable. We do not
    # want to provide an unexpected adapter training experience to customers, so at this
    # point, we will raise an exception in the training process.
    # https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L1149
    raise ValueError("Cannot find a suitable padding token to use, which is mandatory for TRL training.")


def sample_and_split_dataset(
    ds: datasets.DatasetDict,
    train_fraction: float = TRAINING_DEFAULT_DATASET_FRACTION,
    train_test_split: float = TRAINING_DEFAULT_TRAIN_TEST_SPLIT,
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    """
    Optionally downsamples a training dataset, and separates out a train/test split dataset, from a datset dict.
    All sampling will occur WITH shuffling by default and using a constant seed, so that shuffling and downsampling
    can be deterministic across calls to the method. At minimum, the input dataset dict must contain a 'train'
    split.

    If a downsampling of the 'train' dataset is requested through the train_fraction parameter being anything
    other than 1.0, then the 'train' dataset will be downsampled WITH shuffling by default, using a constant seed. If
    there is a 'test' or an 'eval' dataset, then downsampling will NOT occur on this split.

    If a dataset only contains a 'train' split, then the dataset will be split between a 'train' and a 'test' dataset
    by the train_test_split parameter. If a dataset contains a 'test' or an 'eval' split, then the train/test split
    logic does not apply. This logic will run only after the 'train' dataset is downsampled.
    """

    assert isinstance(ds, datasets.DatasetDict)
    assert 'train' in ds

    # If downsampling is requested, then run downsampling on the train dataset.
    ds_train: datasets.Dataset = ds['train']
    if train_fraction < 1.0:
        ds_train = ds_train.train_test_split(
            test_size=1.0 - train_fraction,
            shuffle=True,
            seed=TRAINING_DATASET_SEED)['train']

    # If there is an eval or test split already available in the dataset, then use this as the provided test/eval
    # dataset. If not, perform another dataset split on the 'train' dataset
    # that will split the datset between train and test.
    if 'eval' in ds or 'test' in ds:
        ds_test = ds['eval'] if 'eval' in ds else ds['test']
    else:
        ds_split = ds_train.train_test_split(
            test_size=1.0 - train_test_split,
            shuffle=True,
            seed=TRAINING_DATASET_SEED)
        ds_train, ds_test = ds_split['train'], ds_split['test']

    # Return the final train and test datasets
    return ds_train, ds_test
