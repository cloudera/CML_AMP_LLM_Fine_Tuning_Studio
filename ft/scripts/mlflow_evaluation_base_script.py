import subprocess
import time
from ft.eval.mlflow_driver import driver
import argparse
import os
from ft.client import FineTuningStudioClient

# Function to install required packages

# Start MLflow server as a background process
mlflow_server = subprocess.Popen(["mlflow", "server"])

# Give the MLflow server some time to start
time.sleep(10)  # Adjust this delay as necessary

# Parse arguments from environment variable
arg_string = os.environ.get('JOB_ARGUMENTS', '')

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_id", help="Path of the dataset", default=None)
parser.add_argument("--base_model_id", help="Path of the base model", default=None)
parser.add_argument("--adapter_id", help="Path of the adapter", default=None)
parser.add_argument("--result_dir", help="Path of result dir", required=True)
parser.add_argument("--fts_server_ip", help="IP address of the FTS gRPC server.", required=True)
parser.add_argument("--fts_server_port", help="Exposed port of the gRPC server", required=True)
parser.add_argument("--adapter_bnb_config_id", help="ID of the adapter quantization config", default=None)
parser.add_argument("--model_bnb_config_id", help="ID of the model quantization config", default=None)
parser.add_argument("--generation_config_id", help="ID of the generation config", default=None)


args = parser.parse_args(arg_string.split())

client: FineTuningStudioClient = FineTuningStudioClient(server_ip=args.fts_server_ip, server_port=args.fts_server_port)

try:

    # Execute the evaluation
    response = driver(
        dataset_id=args.dataset_id,
        base_model_id=args.base_model_id,
        adapter_id=args.adapter_id,
        bnb_config_id=args.adapter_bnb_config_id,  # only use bnb config of the adapter for all model layers, for now
        generation_config_id=args.generation_config_id,
        client=client
    )
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
