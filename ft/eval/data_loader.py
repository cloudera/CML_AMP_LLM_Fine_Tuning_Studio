from datasets import load_dataset
import pandas as pd
# from eval.configs import DATASETS, PROMPTS
from ft.eval.utils.template_utils import format_template, extract_eval_column_name, guess_eval_column
from ft.client import FineTuningStudioClient
from ft.api import *


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
        try:
            eval_df = pd.DataFrame(loaded_dataset["test"])
        except BaseException:
            print("There is no test data split present. Hence loading the train split.")
            eval_df = pd.DataFrame(loaded_dataset["train"])
        try:  # if prompt_metadata.input_template:
            eval_prompt_string = prompt_metadata.input_template
            eval_column_name = extract_eval_column_name(prompt_metadata.completion_template)
            eval_df = eval_df.sample(n=total_examples)
            eval_df['model_input'] = eval_df.apply(lambda x: format_template(eval_prompt_string, x), axis=1)
        except BaseException:
            eval_prompt_string = prompt_metadata.prompt_template
            eval_column_name = guess_eval_column(eval_prompt_string)
            eval_df = eval_df.sample(n=total_examples)
            eval_df['model_input'] = eval_df.apply(lambda x: format_template(eval_prompt_string, x), axis=1)

        print(eval_df)
        return eval_df, eval_column_name


if __name__ == "__main__":
    dataloader = Dataloader()
    df, eval_column_name = dataloader.fetch_evaluation_dataset('Clinton/Text-to-sql-v1')
    print(df)
