from datasets import load_dataset
import pandas as pd
# from eval.configs import DATASETS, PROMPTS
from ft.client import FineTuningStudioClient
from ft.api import *
from ft.training.utils import map_dataset_with_prompt_template, split_dataset
from ft.consts import EVAL_INPUT_COLUMN, EVAL_OUTPUT_COLUM, DATASET_FRACTION_THRESHOLD_FOR_EVALUATION
from typing import List
import ast


class Dataloader:

    @staticmethod
    def fetch_evaluation_dataset(
            dataset_id: str,
            total_examples: int = 100,
            client: FineTuningStudioClient = None,
            prompt_metadata=None,
            dataset_split: GetDatasetSplitByAdapterMetadata = None,
            selected_features: List[str] = []):
        dataset: DatasetMetadata = client.GetDataset(GetDatasetRequest(id=dataset_id)).dataset
        if not dataset or dataset == DatasetMetadata():
            # return this as error in UI
            raise ValueError(f"Dataset with id of {dataset_id} not found in the available datasets.")
        # TODO: remove hardcoded dependency on HF name (allow for project-relative dataset loading)
        dataset_fraction = dataset_split.dataset_fraction
        train_test_split = dataset_split.train_test_split
        dataset_hf_name = dataset.huggingface_name
        loaded_dataset = load_dataset(dataset_hf_name)
        if "test" in loaded_dataset:
            loaded_dataset = load_dataset(dataset_hf_name, split="test")
        elif "eval" in loaded_dataset:
            loaded_dataset = load_dataset(dataset_hf_name, split="eval")
        else:
            if int(100 * dataset_fraction) <= DATASET_FRACTION_THRESHOLD_FOR_EVALUATION:
                loaded_dataset = load_dataset(dataset_hf_name, split=f"train[{int(100 * dataset_fraction)}%:]")
            else:
                loaded_dataset = load_dataset(dataset_hf_name, split=f"train[:{int(100 * dataset_fraction)}%]")
                _, loaded_dataset = split_dataset(loaded_dataset, train_test_split)
        # Map both the input and output prompt templates.
        loaded_dataset = map_dataset_with_prompt_template(
            dataset=loaded_dataset,
            prompt_template=prompt_metadata.input_template,
            data_text_field=EVAL_INPUT_COLUMN,
            add_eos_token=False
        )
        loaded_dataset = map_dataset_with_prompt_template(
            dataset=loaded_dataset,
            prompt_template=prompt_metadata.completion_template,
            data_text_field=EVAL_OUTPUT_COLUM,
            add_eos_token=False
        )
        eval_column_name = EVAL_OUTPUT_COLUM

        eval_df = pd.DataFrame(loaded_dataset)
        try:
            print(selected_features)
            selected_features = ast.literal_eval(selected_features)
            if type(selected_features) != list:
                selected_features = []
        except Exception as e:
            print(f"Error parsing selected_features: {e}")
            selected_features = []
        eval_df = eval_df.sample(n=total_examples)
        all_columns_to_be_displayed = selected_features.extend([EVAL_INPUT_COLUMN, EVAL_OUTPUT_COLUM])
        eval_df = eval_df.loc[:, all_columns_to_be_displayed]
        print(eval_df)
        return eval_df, eval_column_name
    


if __name__ == "__main__":
    dataloader = Dataloader()
    df, eval_column_name = dataloader.fetch_evaluation_dataset('Clinton/Text-to-sql-v1')
    print(df)
