from datasets import load_dataset
import pandas as pd
# from eval.configs import DATASETS, PROMPTS
from ft.client import FineTuningStudioClient
from ft.api import *
from ft.training.utils import map_dataset_with_prompt_template
from ft.consts import EVAL_INPUT_COLUMN, EVAL_OUTPUT_COLUM


class Dataloader:

    @staticmethod
    def fetch_evaluation_dataset(
            dataset_id: str,
            total_examples: int = 100,
            client: FineTuningStudioClient = None,
            prompt_metadata=None):
        dataset: DatasetMetadata = client.GetDataset(GetDatasetRequest(id=dataset_id)).dataset
        if not dataset or dataset == DatasetMetadata():
            # return this as error in UI
            raise ValueError(f"Dataset with id of {dataset_id} not found in the available datasets.")
        # TODO: remove hardcoded dependency on HF name (allow for project-relative dataset loading)
        dataset_hf_name = dataset.huggingface_name
        loaded_dataset = load_dataset(dataset_hf_name)

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

        try:
            eval_df = pd.DataFrame(loaded_dataset["test"])
        except BaseException:
            print("There is no test data split present. Hence loading the train split.")
            eval_df = pd.DataFrame(loaded_dataset["train"])

        eval_df = eval_df.sample(n=total_examples)
        eval_df = eval_df.loc[:, [EVAL_INPUT_COLUMN, EVAL_OUTPUT_COLUM]]
        print(eval_df)
        return eval_df, eval_column_name


if __name__ == "__main__":
    dataloader = Dataloader()
    df, eval_column_name = dataloader.fetch_evaluation_dataset('Clinton/Text-to-sql-v1')
    print(df)
