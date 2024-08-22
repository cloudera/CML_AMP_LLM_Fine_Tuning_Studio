from datasets import load_dataset
import pandas as pd
# from eval.configs import DATASETS, PROMPTS
from ft.fine_tune import get_unique_cache_dir
from ft.eval.utils.template_utils import fetch_eval_column_name_and_merge_function
from ft.client import FineTuningStudioClient
from ft.api import *


class Dataloader:

    @staticmethod
    def fetch_evaluation_dataset(dataset_id: str, total_examples: int = 50, client: FineTuningStudioClient = None):
        dataset: DatasetMetadata = client.GetDataset(GetDatasetRequest(id=dataset_id)).dataset
        if not dataset or dataset == DatasetMetadata():
            # return this as error in UI
            raise ValueError(f"Dataset with id of {dataset_id} not found in the available datasets.")
        # TODO: remove hardcoded dependency on HF name (allow for project-relative dataset loading)
        dataset_hf_name = dataset.huggingface_name
        loaded_dataset = load_dataset(dataset_hf_name, cache_dir=get_unique_cache_dir())
        eval_df = pd.DataFrame(loaded_dataset["train"])
        eval_column_name, template_function = fetch_eval_column_name_and_merge_function(dataset_hf_name)
        eval_df = eval_df.sample(n=total_examples)
        eval_df['model_input'] = eval_df.apply(lambda x: template_function(x), axis=1)
        print(eval_df)
        return eval_df, eval_column_name


if __name__ == "__main__":
    dataloader = Dataloader()
    df, eval_column_name = dataloader.fetch_evaluation_dataset('Clinton/Text-to-sql-v1')
    print(df)
