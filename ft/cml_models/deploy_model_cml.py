import cmlapi
import os
import time
from cmlapi import CMLServiceApi
from ft.client import FineTuningStudioClient

FILEPATH = "ft/cml_models/predict.py"


def get_inference_runtime_identifier(cml: CMLServiceApi) -> str:
    """
    Get a runtime ID to be used for inference in CML models. For now, we will use
    the same runtime ID as the fine tuning training jobs, since that runtime ID has
    all components necessary for GPU inference on CUDA enabled devices.
    """
    project_id = os.getenv("CDSW_PROJECT_ID")
    base_job_name = "Finetuning_Base_Job"

    ft_base_job_id = cml.list_jobs(project_id,
                                   search_filter='{"name":"%s"}' % base_job_name).jobs[0].id
    template_job = cml.get_job(
        project_id=project_id,
        job_id=ft_base_job_id
    )
    return template_job.runtime_identifier


def deploy_model(
        model_name: str,
        model_description: str,
        base_model_hf_name: str,
        adapter_location: str,
        fts: FineTuningStudioClient):
    print("Deploying")
    project_id = os.getenv("CDSW_PROJECT_ID")
    # TODO: using base_model_id and adapter_id, override the prediction script.
    client = cmlapi.default_client()
    project: cmlapi.Project = client.get_project(project_id)
    model_body = cmlapi.CreateModelRequest(project_id=project.id, name=model_name, description=model_description)
    model = client.create_model(model_body, project.id)
    model_build_body = cmlapi.CreateModelBuildRequest(
        project_id=project.id,
        model_id=model.id,
        file_path=FILEPATH,
        function_name="api_wrapper",
        runtime_identifier=get_inference_runtime_identifier(client),
        kernel="python3")
    model_build = client.create_model_build(model_build_body, project.id, model.id)
    while model_build.status not in ["built", "build failed"]:
        print("waiting for model to build...")
        time.sleep(10)
        model_build = client.get_model_build(project.id, model.id, model_build.id)

    if model_build.status == "build failed":
        print("model build failed, see UI for more information")
        raise Exception("model build failed, see CML models UI for more information")
    print("model built successfully!")
    model_deployment_body = cmlapi.CreateModelDeploymentRequest(
        project_id=project.id,
        model_id=model.id,
        build_id=model_build.id,
        cpu=2,
        memory=8,
        nvidia_gpus=1,
        environment = {
            "FINE_TUNING_STUDIO_BASE_MODEL_HF_NAME": base_model_hf_name,
            "FINE_TUNING_STUDIO_ADAPTER_LOCATION": adapter_location
        })
    model_deployment = client.create_model_deployment(model_deployment_body, project.id, model.id, model_build.id)
    while model_deployment.status not in ["stopped", "failed", "deployed"]:
        print("waiting for model to deploy...")
        time.sleep(10)
        model_deployment = client.get_model_deployment(project.id, model.id, model_build.id, model_deployment.id)
    if model_deployment.status != "deployed":
        raise Exception("model deployment failed, see UI for more information")
    print("model deployed successfully!")
    return True


if __name__ == "__main__":
    deploy_model()
