import streamlit as st
import os
import json
import requests
from typing import Union
import time 

import cmlapi
from cmlapi import CMLServiceApi
from pgs.streamlit_utils import get_fine_tuning_studio_client
from ft.client import FineTuningStudioClient
from ft.api import *

st.set_page_config(layout="wide")
st.title("📝 Events Ticketing Support Agent")

CML_MODEL_NAME = "Ticketing Agent Model"
HF_MODEL_NAME = "bigscience/bloom-1b1"
ADAPTER_LOCATION = "data/adapters/bloom-1b-ticketing"
MODEL_SERVICE_URL = "https://modelservice." + os.getenv("CDSW_DOMAIN") + "/model"

fts: FineTuningStudioClient = get_fine_tuning_studio_client()
cml: CMLServiceApi = cmlapi.default_client()

with open('examples/ticketing-agent-app/customer-help-library.json') as f:
    customer_help_dict = json.load(f)

def find_customer_help(classification):
  if classification.strip() in customer_help_dict:
    return customer_help_dict[classification.strip()]
  else:
    return "Unknown customer help category [%s]." % classification

if 'status_class_generating' not in st.session_state:
    st.session_state['status_class_generating'] = False

if 'gen_class' not in st.session_state:
    st.session_state['gen_class'] = ""
    
if 'customer_input' not in st.session_state:
    st.session_state['customer_input'] = ""

if 'Suggested Customer Service Actions' not in st.session_state:
    st.session_state['Suggested Customer Service Actions'] = ""
    
if 'help_text' not in st.session_state:
    st.session_state['help_text']= """
        **Examples**:
        
        I need to cancel my concert tickets
        
        Can I get the status of my ticket refund?
        
        Let me talk to an agent

    """


def call_cml_model(customer_input, fts: FineTuningStudioClient, cml: CMLServiceApi):
  
    # Extract out the prompt template that should be used for this adapter
    adapter: AdapterMetadata = list(filter( lambda x: x.location == ADAPTER_LOCATION, fts.get_adapters()))[0]
    prompt: PromptMetadata = fts.stub.GetPrompt(GetPromptRequest(
      id=adapter.prompt_id
    )).prompt
    
    # Create the input prompt
    input_features_dict = {"instruction": customer_input}
    ticketing_model_input = prompt.input_template.format(**input_features_dict)
    ticketing_model_input = ticketing_model_input.replace('\n', '\\n').replace('\t', '\\t')
    response = requests.post(MODEL_SERVICE_URL + f"?accessKey={get_ticketing_model(cml).access_key}",
                      data=json.dumps({"request":{"prompt":ticketing_model_input}}),
                      headers={'Content-Type': 'application/json', 'Authorization': 'Bearer %s' % os.getenv('CDSW_APIV2_KEY')})

    response_dict = response.json()
    if 'success' in response_dict:
      print(response_dict)
      return response_dict['response'].strip()
    elif 'errors' in response_dict:
      return "ERROR: " + json.dumps(response_dict['errors'])
    else:
      return "ERROR: Check application logs"

def deploy_ticketing_cml_model(fts: FineTuningStudioClient) -> ExportModelResponse:
  """
  Deploy the ticketing model that is required for this example as a CML model.
  """
  
  model: ModelMetadata = list(filter(lambda x: x.huggingface_model_name == HF_MODEL_NAME , fts.get_models()))[0]
  adapter: AdapterMetadata = list(filter(lambda x: x.location == ADAPTER_LOCATION , fts.get_adapters()))[0]
  return fts.stub.ExportModel(
    ExportModelRequest(
      type=ModelExportType.CML_MODEL,
      base_model_id=model.id,
      adapter_id=adapter.id,
      model_name=CML_MODEL_NAME,
      model_description=CML_MODEL_NAME
    )
  )

def get_ticketing_model(cml: CMLServiceApi) -> Union[cmlapi.Model, None]:
  """
  Check to see if the ticketing model exists as a CML model.
  """
  resp: cmlapi.ListModelsResponse = cml.list_models(os.getenv("CDSW_PROJECT_ID"))
  models: list[cmlapi.Model] = resp.models
  models = list(filter(lambda x: x.name == CML_MODEL_NAME, models))
  for model in models:
    if model.deletion_status == '':
      return model 
  return None


def is_ticketing_model_deployed(cml: CMLServiceApi) -> bool:
  """
  Check to see if the ticketing model has been deployed.
  """
  
  model: cmlapi.Model = get_ticketing_model(cml)
  if not model:
    return False

  mbresp: cmlapi.ListModelBuildsResponse = cml.list_model_builds(
    project_id=os.getenv("CDSW_PROJECT_ID"),
    model_id=model.id
  )
  model_builds: list[cmlapi.ModelBuild] = mbresp.model_builds
  if not model_builds:
    return False
  build_times = [x.updated_at for x in model_builds]
  model_build: cmlapi.ModelBuild = list(filter(lambda x: x.updated_at == max(build_times), model_builds))[0]
  
  resp: cmlapi.ListModelDeploymentsResponse = cml.list_model_deployments(
    project_id=os.getenv("CDSW_PROJECT_ID"),
    model_id=model.id,
    build_id=model_build.id
  )
  model_deployments: list[cmlapi.ModelDeployment] = resp.model_deployments
  if not model_deployments:
    return False
  deployment_times = [x.updated_at for x in model_deployments]
  
  # Sort by maximum model deployment ID
  model_deployment: cmlapi.ModelDeployment = list(filter(lambda x: x.updated_at == max(deployment_times), model_deployments))[0]
  
  return True if model_deployment.status == "deployed" else False

def render_app_landing_page():
  st.header("The model to run this app is not yet deployed.")
  st.text(f"This sample application requires the \"{CML_MODEL_NAME}\" model. Deploy this model now to start using the app!")
  deploy_model = st.button(f"Deploy \"{CML_MODEL_NAME}\"", type="primary",)
  if deploy_model:
    deploy_ticketing_cml_model(fts)
    time.sleep(2)
    st.rerun()

def render_ticketing_app(fts: FineTuningStudioClient, cml: CMLServiceApi):
  col1, col2= st.columns([1,1])
  with col1:
    class_container = st.container(border=True)
    with class_container:
      st.markdown("### Customer Input")
      st.caption("Customer Support Request Text")
      st.session_state['customer_input'] = st.text_input(
          "",
          placeholder="Insert sample customer input here...",
          disabled=st.session_state['status_class_generating'],
          value=st.session_state['customer_input'],
          label_visibility="collapsed"
      )
      if st.button("Submit",
               key='class_btn',
              disabled= st.session_state['status_class_generating']):
        # Perform inference using the loaded model
        with st.spinner('Generating...'):
          st.session_state['gen_class'] = call_cml_model(st.session_state['customer_input'], fts, cml)
          st.session_state['Suggested Customer Service Actions'] = find_customer_help(st.session_state['gen_class'])
      st.markdown(st.session_state['help_text'])
      
  with col2:
    suggestion_container = st.container(border=True)
    with suggestion_container:
      st.markdown("""
      ### Support Agent Response
      """)
      class_txt = st.container(border=True)
      st.caption("Detected Customer Intent")
      st.code(body=st.session_state['gen_class'])
      st.caption("Suggested Response to Customer")
      suggestion_txt = st.container(border=True)
      suggestion_txt.write(st.session_state['Suggested Customer Service Actions'])  


if is_ticketing_model_deployed(cml):
  render_ticketing_app(fts, cml)
elif get_ticketing_model(cml):
  st.header("Sample application model is being deployed...")
  st.text(f"Please refresh the page when the model \"{CML_MODEL_NAME}\" reaches \"deployed\" status.")
else:
  render_app_landing_page()

