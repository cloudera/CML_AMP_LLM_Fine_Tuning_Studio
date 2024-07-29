import subprocess
import time
import pandas as pd
from glob import glob
from ft.eval.mlflow_driver import driver
from ft.eval.eval_job import StartEvaluationRequest
import argparse
import os

# Function to install required packages

# Start MLflow server as a background process
mlflow_server = subprocess.Popen(["mlflow", "server"])

# Give the MLflow server some time to start
time.sleep(10)  # Adjust this delay as necessary

# Parse arguments from environment variable
arg_string = os.environ.get('JOB_ARGUMENTS', '')

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", help="Path of the dataset")
parser.add_argument("--basemodel", help="Path of the base model")
parser.add_argument("--adapter_path", help="Path of the adapter")
parser.add_argument("--result_dir", help="Path of result dir", required=True)

args = parser.parse_args(arg_string.split())

try:
    # Create the evaluation request
    request = StartEvaluationRequest(
        adapter_path=args.adapter_path,
        base_model_name=args.basemodel,
        dataset_name=args.dataset
    )

    # Execute the evaluation
    response = driver(request)
    print(response.metrics)
    print(response.csv)

    # Save the CSV result to a file in the result directory
    result_dir = args.result_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    file_name = os.path.join(result_dir, "result_evaluation.csv")
    response.csv.to_csv(file_name, encoding='utf-8')

finally:
    # Stop the MLflow server
    mlflow_server.terminate()
    mlflow_server.wait()
