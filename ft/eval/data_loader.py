from ft.datasets import load_dataset_into_memory
import pandas as pd
# from eval.configs import DATASETS, PROMPTS
from ft.client import FineTuningStudioClient
from ft.api import *
from ft.training.utils import map_dataset_with_prompt_template, sample_and_split_dataset
from ft.consts import EVAL_INPUT_COLUMN, EVAL_OUTPUT_COLUM
from typing import List
import ast
import datasets


class Dataloader:

    @staticmethod
    def fetch_evaluation_dataset(
            dataset_id: str,
            total_examples: int = 100,
            client: FineTuningStudioClient = None,
            prompt_metadata=None,
            dataset_split: GetDatasetSplitByAdapterMetadata = None,
            selected_features: List[str] = [],
            eval_dataset_fraction: float = 1.0):
        dataset: DatasetMetadata = client.GetDataset(GetDatasetRequest(id=dataset_id)).dataset
        if not dataset or dataset == DatasetMetadata():
            # return this as error in UI
            raise ValueError(f"Dataset with id of {dataset_id} not found in the available datasets.")

        dataset_fraction = dataset_split.dataset_fraction
        train_test_split = dataset_split.train_test_split

        # Load a generic dataset into memroy
        loaded_dataset: datasets.DatasetDict = load_dataset_into_memory(dataset)

        # Extract out the dataset split that corresponds to test/eval. If there is a 'test' or 'eval'
        # split already available in the dataset, this is used. If not, then the 'train' split is split
        # into a new train/test split based on how the adapter was trained (if trained with Studio).
        # Dataset splitting is deterministic, so we can ensure that we are only evaluating on data
        # that was not presented in the training process.
        _, ds_test = sample_and_split_dataset(
            loaded_dataset,
            train_fraction=dataset_fraction,
            train_test_split=train_test_split
        )

        # Map both the input and output prompt templates.
        ds_test = map_dataset_with_prompt_template(
            dataset=ds_test,
            prompt_template=prompt_metadata.input_template,
            data_text_field=EVAL_INPUT_COLUMN,
            add_eos_token=False
        )
        ds_test = map_dataset_with_prompt_template(
            dataset=ds_test,
            prompt_template=prompt_metadata.completion_template,
            data_text_field=EVAL_OUTPUT_COLUM,
            add_eos_token=False
        )
        eval_column_name = EVAL_OUTPUT_COLUM

        eval_df = pd.DataFrame(ds_test)
        try:
            print(selected_features)
            selected_features = ast.literal_eval(selected_features)
            if not isinstance(selected_features, list):
                selected_features = []
        except Exception as e:
            print(f"Error parsing selected_features: {e}")
            selected_features = []
        eval_df = eval_df.sample(frac=eval_dataset_fraction)
        selected_features.extend([EVAL_INPUT_COLUMN, EVAL_OUTPUT_COLUM])
        eval_df = eval_df.loc[:, selected_features]
        print(eval_df)
        return eval_df, eval_column_name


if __name__ == "__main__":
    dataloader = Dataloader()
    df, eval_column_name = dataloader.fetch_evaluation_dataset('Clinton/Text-to-sql-v1')
    print(df)
