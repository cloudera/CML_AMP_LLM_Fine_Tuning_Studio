# CML Fine Tuning Studio: Technical Overview

## Scope

This document outlines the technical components of the CML Fine Tuning Studio product. This is designed to be read by users of the tool that want deeper understanding of the framework behind the tool, or developers who are onboarding to the tool’s architecture and who would like to extend features in the AMP.

For users who just want to interact with the vanilla application, please see the [User Guide](user_guide.md).

## Introduction

The CML Fine Tuning Studio (will shorthand as the “Studio” through the document) is a Cloudera-developed AMP that provides users with an all-encompassing application and “ecosystem” for managing, fine tuning, and evaluating LLMs. This application is a “launcher” that helps users organize and dispatch other CML Workloads (primarily CML Jobs) that are configured specifically for LLM training and evaluation type tasks.

The AMP itself consists of multiple different components:

* CML Application designed using [Streamlit](https://streamlit.io/)   
* A python library, `ft`, for managing the fine tuning application, including:  
  * An application manager that handles requests, dispatches jobs  
  * A set of [pydantic models](https://docs.pydantic.dev/latest/concepts/models/) to represent data structure of requests  
  * Utility classes and functions for managing CML Jobs/environments  
* Template CML Job definitions for 1\) fine tuning and 2\) evaluation  
  * These are template scripts that utilize the `ft` package  
* A lightweight project-relative metadata store for saving application state between browser sessions

High-level architecture:


![CML Fine Tuning Studio AMP](../resources/images/fts_architecture.png)


## Project Security and Scope

When deploying this AMP to a user’s workspace, the application exists primarily within the scope of the a user’s project that is created for the AMP. The project is created as a **private** project by default, which means that other CML users within the CML workspace will not be able to access your specific instance of the application unless the application is shared with another user. If the application is shared with another user, this application does not claim any concurrency promises when multiple different users (multiple different browser sessions) are manipulating project files at the same time. There are a few components that this application interacts with that don’t explicitly live within the scope of a CML project:

* When deploying models to the CML model registry, models are deployed to a workspace-scoped model registry. If a user does not have access to the workspace’s Model Registry, then Model Registry import/export features will not work.  
* When tracking models with MLFlow, model artifacts and metrics are sent to the workspace’s MLFlow tracking server. As with Model Registry, this is scoped to whatever CML Experiments/MLFLow permissions the user has available/

## Streamlit UI

This AMP utilizes [Streamlit](https://streamlit.io/) for UI design. The entrypoint for the Streamlit app can be found in `main.py`, and the CML Application entrypoint script can be found in `bin/run-app.py`. The actual Streamlit app itself is served on the `CDSW_APP_PORT` and is available at the specified application’s domain.

```py
!streamlit run main.py --server.port $CDSW_APP_PORT --server.address 127.0.0.1
```

The Streamlit “pages” (frontend UI components) are all stored in the `pgs/` directory of the AMP. All of the pages in this application contain some logic that interacts with the application’s backend (also referred to as the application “instance” in some contexts), and potentially logic that interacts with the application’s state. If modifying or adding Streamlit UI pages to this application, you can technically perform any operations you want (for example, call `cmlapi` commands and run complex logic directly within the page), but it’s recommended for development purposes that developers route all requests through the `FineTuningApp` instance (talked about in the next section) and then route commands through the instance via `ft.app.get_app()`.

Streamlit is not a typical frontend UI framework. Because it was designed for ease-of-use and simplicity for data science projects, there are some limitations. For example, every interaction within the UI will lead to a complete re-rendering of the whole page (unless a user annotates certain sections of a page with a `@streamlit.fragment` decorator). If working with Streamlit components, it’s highly recommended to read through [Streamlit’s execution model](https://docs.streamlit.io/develop/concepts/architecture) for better understanding on how Streamlit works.

## Application Instance & Application Managers

In its current form, the Application that this AMP deploys will instantiate a singleton instance of the `FineTuningApp()` class. This class acts as the API facade for interacting with backend components of the application. In the future, the team is looking into implementing protobufs and a gRPC client for interacting with application logic, which would replace the need for pydantic data models (our data models will be generated automatically), and would replace the need for gRPC application client management (automatic client interfaces can be generated).

The application instance can be accessed globally via helper functions:

```py
import ft
from ft.app import FineTuningApp
from ft.model import ImportModelRequest, ImportModelResponse

# Get an instance to the app
app: FineTuningApp = ft.app.get_app()

# Perform API operations on the app
resp: ImportModelResponse = app.import_model(ImportModelRequest(...))
```

### Application Managers

To help simplify the API surface, most of the intense application logic resides in *managers* that exist in the `FineTuningApp` class. These managers are the sub-classes that are responsible for actually interacting with components of the application. Several of the API features available in the `FineTuningApp` are actually just calling the equivalent sub-class method, but wraps some controllers around the application session state (talked about in the next section). 

```py
"""
========================
ft/app.py:
========================
"""

class FineTuningApp():
	
    models: ModelsManagerBase
    ...

    def __init__(self, props: FineTuningAppProps):
        self.models = props.models_manager
        ...

    def import_model(self, request: ImportModelRequest) -> ImportModelResponse:

        # Call the sub-class method
        import_response: ImportModelResponse = self.models.import_model(request)

        # If we've successfully imported a new model, then make sure we update
        # the app's model state with this data.
        if import_response.model is not None:
            state: AppState = get_state()
            models: List[ModelMetadata] = state.models
            models.append(import_response.model)
            update_state({"models": models})

        return import_response


"""
========================
Within a streamlit page:
========================
"""

# Interacting with the app:
app: FineTuningApp = ft.app.get_app()

# Perform API operations on the top-level app function, not the lower-level.
resp: ImportModelResponse = app.import_model(ImportModelRequest(...))
```

When instantiating a `FineTuningApp` object, a list of managers need to be passed to a `FineTuningAppProps` object, such as:

```py
from ft.managers.models import ModelsManagerSimple
from ft.managers.datasets import DatasetsManagersSimple
from ft.managers.jobs import FineTuningJobsManagerSimple
from ft.managers.mlflow import MLFlowEvaluationJobsManagerSimple

app: FineTuningApp = FineTuningApp(
    FineTuningAppProps(
        datasets_manager=DatasetsManagerSimple(),
        models_manager=ModelsManagerSimple(),
        jobs_manager=FineTuningJobsManagerSimple(),
        mlflow_manager=MLflowEvaluationJobsManagerSimple()
    )
)
```

The reason for this design is to enable developers to build “mock” managers, “test” managers, or other types of managers that require specific plugins. This dependency-injection-type structure helps with things like designing tests too, where a test fixture can construct a fake `FineTuningApp` with mock managers. This is also convenient if a user wants to just test out new Streamlit UI features without fully deploying to a CML workspace (i.e., developing with limited or no access to `cmlapi` operations).

```py
from ft.managers.models import ModelManagerBase
from ft.models import (
    ModelMetadata,
    ImportModelRequest,
    ImportModelResponse,
    ExportModelRequest,
    ExportModelResponse
)
import ft
...


"""
=============================
Mock model manager definition
=============================
"""

class MyMockModelManager(ModelManagerBase):
    @override
    def list_models(self) -> List[ModelMetadata]:
        ...
    @override
    def import_model(self, request: ImportModelRequest) -> ImportModelResponse:
        ...
    @override
    def export_model(self, request: ExportModelRequest) -> ExportModelResponse:
        ...

"""
====================================================
in streamlit entrypoint file (main.py or otherwise):
====================================================
"""

ft.app.create_app(
    FineTuningAppProps(
        models_manager=MyMockModelManager(),
        ...
    )
)
```

> **NOTE**: this component of the framework may change if we move from a pydantic/singleton framework to a protobuf/gRPC framework. 

## Application Session State

To store data for a user in-between browser sessions, this AMP maintains a lightweight metadata/session state file that contains metadata about the datasets, models, prompts, training jobs, and evaluation jobs within a project that all tie to this specific project instance. Note that this application state file is only designed for metadata. Actual binary data (model checkpoints, tokenizers, dataset data, etc.) is not stored in the session state. Instead, CML Jobs for fine tuning and evaluation read from the application’s session store file to extract model metadata, and then download model data based on the metadata in the session state. In this way, the AMP can limit the amount of information passed to workloads. For example, for a fine tuning job, we only need to send the unique ID of a model as part of the fine tuning request object. The CML workload that spins up for the fine tuning job can then access the application’s session store to extract metadata based on this model ID, and download the model to memory based on the model metadata.

The metadata session store location is defined by the `FINE_TUNING_APP_STATE_LOCATION` environment variable that is configured at initialization of the AMP. By default, this file is `.app/state.json` and can be found in a project’s Files (you may need to enable the `Show hidden files` option in the CML project file browser).

### Session State Format and Access

A default session state is provided in this AMP’s repository at the default `.app/state.json` location with some default datasets, models, and prompts, in order to guide first-time users through the capabilities of the application. Each item in the session state (including the entirety of the session state file) is controlled by a pydantic model type that is defined in the `ft` package. The `ft` package also provides helper functions to extract state information and write state information to the session state file. When working on custom modifications to this AMP, a developer can access state information in the project as follows:

```py
import ft
from ft.state import AppState
from ft.model import ModelMetadata, ModelType
from typing import List
from transformers import AutoModelForCausalLM

# Extract the current state from the model session state file
current_state: AppState = ft.state.get_state()

# Perform operations on a model (for example, loading a model)
model_id = "placeholder_model_id"
models: List[ModelMetadata] = current_state.models
model_md: ModelMetadata = filter(lambda x: x.id == model_id, models)[0]
model_type: ModelType = model_md.type

# Load the model
model = None
if model_type == ModelType.HUGGINGFACE:
    model = AutoModelForCausalLM.from_pretrained(model_md.huggingface_model_name)
```
