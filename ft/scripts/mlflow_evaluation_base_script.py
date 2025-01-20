from ft.venv_utils import activate_venv
activate_venv(".venv")

from ft.eval.mlflow_driver import driver
import argparse
import os
import ast
from ft.client import FineTuningStudioClient
import pandas as pd
from copy import deepcopy
import json
from ft.consts import EVAL_DATASET_DEFAULT_FRACTION, USER_DEFINED_IDENTIFIER, DEFAULT_BNB_CONFIG, DEFAULT_GENERATIONAL_CONFIG


# Parse arguments from environment variable
arg_string = os.environ.get('JOB_ARGUMENTS', '')

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_id", help="Dataset ID to use", default=None)
parser.add_argument("--base_model_id", help="Base model ID to use.", default=None)
parser.add_argument("--adapter_id", help="Adapter ID to use", default=None)
parser.add_argument("--prompt_id", help="Prompt ID to use", default=None)
parser.add_argument("--result_dir", help="Path of result dir", required=True)
parser.add_argument("--adapter_bnb_config", help="adapter quantization config", default=json.dumps(DEFAULT_BNB_CONFIG))
parser.add_argument("--model_bnb_config", help="model quantization config", default=json.dumps(DEFAULT_BNB_CONFIG))
parser.add_argument("--generation_config", help="generation config", default=json.dumps(DEFAULT_GENERATIONAL_CONFIG))
parser.add_argument("--selected_features", help="Names of the columns to be shown in the evaluation csv", default=None)
parser.add_argument("--eval_dataset_fraction", type=float, default=EVAL_DATASET_DEFAULT_FRACTION,
                    help="Percentage of eval dataset to be used for evaluation")
parser.add_argument("--comparison_adapter_id", help="ID of the adapter to be compared", default=None)
parser.add_argument("--job_id", help="UUID of the job", default=None)
parser.add_argument("--run_number", type=int, help="Index of the job to run", default=None)

args = parser.parse_args(arg_string.split())

client: FineTuningStudioClient = FineTuningStudioClient()


# Execute the evaluation
response = driver(
    dataset_id=args.dataset_id,
    base_model_id=args.base_model_id,
    adapter_id=args.adapter_id,
    prompt_id=args.prompt_id,
    bnb_config=args.adapter_bnb_config,  # only use bnb config of the adapter for all model layers, for now
    generation_config=args.generation_config,
    selected_features=args.selected_features,
    eval_dataset_fraction=args.eval_dataset_fraction,
    comparison_adapter_id=args.comparison_adapter_id,
    job_id=args.job_id,
    run_number=args.run_number,
    client=client
)
print(response.metrics)
print(response.csv)

# Save the CSV result to a file in the result directory
result_dir = args.result_dir
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
mod_metrics = deepcopy(response.metrics)

# uncomment the below code to keep only mean

for k, v in response.metrics.items():
    if "mean" not in k:
        del mod_metrics[k]

if "exact_match/v1" in response.metrics:
    mod_metrics["exact_match/v1"] = response.metrics["exact_match/v1"]
f_map = {}
selected_features = ast.literal_eval(args.selected_features)
for feature in selected_features:
    f_map[feature] = f"{feature}{USER_DEFINED_IDENTIFIER}"
response.csv.rename(columns=f_map, inplace=True)
aggregated_results = pd.DataFrame(mod_metrics.items(), columns=["metric", "score"])

file_name = os.path.join(result_dir, "result_evaluation.csv")
response.csv.to_csv(file_name, encoding='utf-8', index=False)

aggregated_file_name = os.path.join(result_dir, "aggregregated_results.csv")
aggregated_results.to_csv(aggregated_file_name, encoding='utf-8', index=False)
