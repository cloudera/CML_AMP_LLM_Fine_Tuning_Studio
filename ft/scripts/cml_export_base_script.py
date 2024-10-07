from ft.cml_models.deploy_model_cml import deploy_model
import argparse
import os
from ft.client import FineTuningStudioClient
from ft.cml_model_export import update_export_job_status
from ft.consts import DeploymentStates

# Function to install required packages


# Parse arguments from environment variable
arg_string = os.environ.get('JOB_ARGUMENTS', '')

parser = argparse.ArgumentParser()

parser.add_argument("--id", help="Identifier for the primary key in the export jobs table", default=None)
parser.add_argument("--base_model_id", help="Base model ID to use.", default=None)
parser.add_argument("--adapter_id", help="Adapter ID to use", default=None)
parser.add_argument("--model_name", help="Name of the model to be exported in CML", default=None)
parser.add_argument("--model_description", help="Description of the model to be exported in CML", default=None)

args = parser.parse_args(arg_string.split())

client: FineTuningStudioClient = FineTuningStudioClient()

try:
    # Deploy the model
    success = deploy_model(
        base_model_id=args.base_model_id,
        adapter_id=args.adapter_id,
        model_name=args.model_name,
        model_description=args.model_description,
        fts=client
    )
    if success:
        print("Model deployed successfully in CML")
        update_export_job_status(args.id, DeploymentStates.SUCCESS.name, client)


except Exception as e:
    print(f"Error deploying model: {e}")
    update_export_job_status(args.id, DeploymentStates.FAILED.name, client)
    # TODO: Update the status of the run as failed in the database. Check for update on nonkey
    raise e
