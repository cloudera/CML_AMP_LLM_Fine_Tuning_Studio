from eval.mlflow_evaluator import ModelEvaluator
from eval.mlflow_logger import ModelLogger
from eval.data_loader import Dataloader
from eval.make_peft_pipeline import fetch_pipeline
import pandas as pd


def driver(dataset_name, base_model_name, adapter_name, device="gpu"):
    dataloader = Dataloader()
    logger = ModelLogger()
    evaluator = ModelEvaluator()

    # Load dataset
    eval_dataset, eval_column_name = dataloader.fetch_evaluation_dataset(dataset_name)
    # Load Model Pipeline
    pipeline = fetch_pipeline(base_model_name, adapter_name, device=device)

    # Log model to MLFlow
    model_info = logger.log_model(pipeline)

    # Evaluate model
    results = evaluator.evaluate_model(model_info, eval_dataset, eval_column_name)

    results_df = pd.DataFrame(results.tables['eval_results_table'])
    return results.metrics, results_df


if __name__ == "__main__":
    results, results_df = driver('philschmid/sql-create-context-copy', 'bigscience/bloom-1b1',
                                 'data/adapters/bloom1b1-lora-sql')
    print(results)
    print(results_df)
