# Ticketing Agent Application


## CML Workloads
This sample application is composed of two CML Workloads
- CML Model for running Mistral-instruct-7b with ticket classifier adapter
  - `examples/ticketing-agent-app/mistral-cml-model.py`
  - By default, this CML Model workload is created and deployed with 
the AMP startup steps
- CML Application for serving a UI Web Application
  - `examples/ticketing-agent-app/ticketing-agent-app.py`

## Files
### `./adapter/mistral-7b-ticketing`
- A copy of the finetuned LoRA adapter `data/adapters/mistral-7b-ticketing`, placed in this directory for convenience.

### `./mistral-cml-model.py`
- Python file containing LLM Model + Adapter loading and inference function definitions

### `ticketing-agent-app.py`
- Python file containing streamlit implementation of a Sample Ticketing Agent application

### `ticketing-agent-launch.py`
- Python file used as CML Application launch warpper script for `streamlit ticketing-agent-app.py`

## Application Recreation
If you need to recreate this sample application follow the steps below:
- Navigate to Applications
- Click `New Application`
- Give it name `Sample-Ticketing-Agent-Application`
- Set Subdomain `sample-ticketing-agent-asdxc`
- Select File `examples/ticketing-agent-app/ticketing-agent-launch.py`
- Pick Runtime
  - PBJ Workbench -- Python 3.9 -- Nvidia GPU -- 2024.05 (or newer)
- Set Resource Profile
  - At least 2CPU / 4MEM
- Set Environment Variables
  - TICKETING_MODEL_ACCESS_KEY : \<access key for sample ticketing agent model\>
- Click `Create Application`


## Model Recreation
If you need to recreate this CML Model deployment follow the steps below:
- Navigate to  Model Deployments
- Click `New Model`
- Give it a Name and Description
- Select File `examples/ticketing-agent-app/mistral-cml-model.py`
- Set Function Name `api_wrapper`
- Pick Runtime
  - PBJ Workbench -- Python 3.9 -- Nvidia GPU -- 2024.05 (or newer)
- Set Resource Profile
  - At least 2CPU / 8MEM
  - 1 GPU
- Click `Deploy Model`
- Wait until it is Deployed
