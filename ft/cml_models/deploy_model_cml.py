import cmlapi
import os
import time

FILEPATH="/home/cdsw/ft/cml_models/predict.py"


def deploy_model():
    print("Deploying")
    project_id = os.getenv("CDSW_PROJECT_ID")
    client = cmlapi.default_client()
    project: cmlapi.Project = client.get_project(project_id)
    model_body = cmlapi.CreateModelRequest(project_id=project.id, name="Demo Model", description="A simple model")
    model = client.create_model(model_body, project.id)
    model_build_body = cmlapi.CreateModelBuildRequest(project_id=project.id, model_id=model.id, file_path=FILEPATH, function_name="api_wrapper", kernel="python3")
    model_build = client.create_model_build(model_build_body, project.id, model.id)
    while model_build.status not in ["built", "build failed"]:
        print("waiting for model to build...")
        time.sleep(10)
        model_build = client.get_model_build(project.id, model.id, model_build.id)

    if model_build.status == "build failed":
        print("model build failed, see UI for more information")
        raise Exception.exit(1)
    print("model built successfully!")
    model_deployment_body = cmlapi.CreateModelDeploymentRequest(project_id=project.id, model_id=model.id, build_id=model_build.id)
    model_deployment = client.create_model_deployment(model_deployment_body, project.id, model.id, model_build.id)
    while model_deployment.status not in ["stopped", "failed", "deployed"]:
        print("waiting for model to deploy...")
        time.sleep(10)
        model_deployment = client.get_model_deployment(project.id, model.id, model_build.id, model_deployment.id)
    if model_deployment.status != "deployed":
        raise Exception("model deployment failed, see UI for more information")
    print("model deployed successfully!")


if __name__ == "__main__":
    deploy_model()