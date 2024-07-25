from datasets import load_dataset
import pandas as pd
#from eval.configs import DATASETS, PROMPTS
from ft.state import get_state
from ft.fine_tune import get_unique_cache_dir
from eval.utils.template_utils import fetch_eval_column_name_and_merge_function


class Dataloader():

    @staticmethod
    def fetch_evaluation_dataset(dataset_name: str, total_examples: int = 50):
        found = False
        dataset_id = None
        DATASETS = get_state().datasets
        for dataset in DATASETS:
            if dataset.huggingface_name == dataset_name:
                dataset_id = dataset.id
                found = True
                break
        if not found:
            # return this as error in UI
            raise ValueError(f"Dataset {dataset_name} not found in the available datasets.")
        dataset = load_dataset(dataset_name, cache_dir=get_unique_cache_dir())
        eval_df = pd.DataFrame(dataset["train"])
        eval_column_name, template_function = fetch_eval_column_name_and_merge_function(dataset_id)
        eval_df = eval_df.sample(n=total_examples)
        eval_df['model_input'] = eval_df.apply(lambda x: template_function(x), axis=1)
        return eval_df, eval_column_name


if __name__ == "__main__":
    dataloader = Dataloader()
    df, eval_column_name = dataloader.fetch_evaluation_dataset('Clinton/Text-to-sql-v1')
    print(df)
