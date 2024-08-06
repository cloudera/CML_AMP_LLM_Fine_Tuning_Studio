from typing import List
from unittest.mock import patch
from uuid import uuid4
import pytest


from ft.api import *
from ft.api import ImportDatasetRequest, ImportDatasetResponse, ImportModelRequest, ImportModelResponse
from ft.managers.models import ModelsManagerBase, ModelsManagerSimple
from ft.managers.datasets import DatasetsManagerBase, DatasetsManagerSimple
from ft.managers.jobs import FineTuningJobsManagerBase, FineTuningJobsManagerSimple
from ft.managers.evaluation import MLflowEvaluationJobsManagerSimple

from ft.app import (
    FineTuningApp,
    FineTuningAppProps,
    ImportDatasetRequest,
    ImportDatasetResponse
)

from tests.mock import *


def generate_test_app(
    models: ModelsManagerBase = MockModelsManager(),
    datasets: DatasetsManagerBase = MockDatasetsManager(),
    jobs: FineTuningJobsManagerBase = MockFineTuningJobsManager(),
    mlflow: MLflowEvaluationJobsManagerBase = MockEvaluatorManager()
) -> FineTuningApp:
    return FineTuningApp(FineTuningAppProps(
        datasets,
        models,
        jobs,
        mlflow
    ))


class TestAppDatasets():

    @patch("ft.app.write_state")
    @patch("ft.app.get_state")
    def test_add_dataset_success(self, get_state, write_state):
        dataset = DatasetMetadata(
            id=str(uuid4()),
            type=DatasetType.DATASET_TYPE_HUGGINGFACE
        )

        class MockDatasetsManagerConstant(MockDatasetsManager):
            def import_dataset(self, request: ImportDatasetRequest) -> ImportDatasetResponse:
                return ImportDatasetResponse(
                    dataset=dataset
                )
        app = generate_test_app(datasets=MockDatasetsManagerConstant())
        get_state.return_value = AppState()
        app.add_dataset(ImportDatasetRequest())
        write_state.assert_called_with(AppState(
            datasets=[
                dataset
            ]
        ))

    @patch("ft.app.write_state")
    @patch("ft.app.get_state")
    def test_add_dataset_no_response(self, get_state, write_state):
        class MockDatasetsManagerConstant(MockDatasetsManager):
            def import_dataset(self, request: ImportDatasetRequest) -> ImportDatasetResponse:
                return ImportDatasetResponse()
        app = generate_test_app(datasets=MockDatasetsManagerConstant())
        get_state.return_value = AppState()
        app.add_dataset(ImportDatasetRequest())
        write_state.assert_not_called()

    @patch("ft.app.write_state")
    @patch("ft.app.get_state")
    def test_remove_dataset_no_prompts(self, get_state, write_state):
        datasets: List[DatasetMetadata] = [
            DatasetMetadata(
                id=str(uuid4()),
                type=DatasetType.DATASET_TYPE_HUGGINGFACE
            )
        ]
        app = generate_test_app()
        get_state.return_value = AppState(datasets=datasets)
        app.remove_dataset(datasets[0].id)
        write_state.assert_called_with(AppState())

    @patch("ft.app.write_state")
    @patch("ft.app.get_state")
    def test_remove_dataset_associated_prompt(self, get_state, write_state):
        datasets: List[DatasetMetadata] = [
            DatasetMetadata(
                id=str(uuid4()),
                type=DatasetType.DATASET_TYPE_HUGGINGFACE
            ),
            DatasetMetadata(
                id=str(uuid4()),
                type=DatasetType.DATASET_TYPE_HUGGINGFACE
            )
        ]

        prompts: List[PromptMetadata] = [
            PromptMetadata(
                id=str(uuid4()),
                name="",
                dataset_id=datasets[0].id,
                prompt_template=""
            ),
            PromptMetadata(
                id=str(uuid4()),
                name="",
                dataset_id=datasets[1].id,
                prompt_template=""
            )
        ]

        get_state.return_value = AppState(
            datasets=datasets,
            prompts=prompts
        )
        app = generate_test_app()
        app.remove_dataset(datasets[0].id)
        write_state.assert_called_with(AppState(
            datasets=[datasets[1]],
            prompts=[prompts[1]]
        ))


class TestAppPrompts():

    @patch("ft.app.write_state")
    @patch("ft.app.get_state")
    def test_add_prompt(self, get_state, write_state):
        prompt = PromptMetadata(
            id=str(uuid4()),
            name="",
            dataset_id=str(uuid4()),
            prompt_template="hello"
        )
        app = generate_test_app()
        get_state.return_value = AppState()
        app.add_prompt(prompt)
        write_state.assert_called_with(AppState(
            prompts=[prompt]
        ))

    @patch("ft.app.write_state")
    @patch("ft.app.get_state")
    def test_remove_prompt(self, get_state, write_state):
        prompt = PromptMetadata(
            id=str(uuid4()),
            name="",
            dataset_id=str(uuid4()),
            prompt_template="hello"
        )
        app = generate_test_app()
        get_state.return_value = AppState(
            prompts=[prompt]
        )
        app.remove_prompt(prompt.id)
        write_state.assert_called_with(AppState())


class TestAppModels():

    @patch("ft.app.write_state")
    @patch("ft.app.get_state")
    def test_import_model_success(self, get_state, write_state):
        model = ModelMetadata(
            id=str(uuid4()),
            type=ModelType.MODEL_TYPE_HUGGINGFACE
        )

        class MockModelsManagerConstant(MockModelsManager):
            def import_model(self, request: ImportModelRequest) -> ImportModelResponse:
                return ImportModelResponse(model=model)
        app = generate_test_app(models=MockModelsManagerConstant())
        get_state.return_value = AppState()
        app.import_model(ImportModelRequest())
        write_state.assert_called_with(AppState(
            models=[
                model
            ]
        ))

    @patch("ft.app.write_state")
    @patch("ft.app.get_state")
    def test_import_model_no_response(self, get_state, write_state):
        class MockModelsManagerConstant(MockModelsManager):
            def import_model(self, request: ImportModelRequest) -> ImportModelResponse:
                return ImportModelResponse()
        app = generate_test_app(models=MockModelsManagerConstant())
        get_state.return_value = AppState()
        app.import_model(ImportModelRequest())
        write_state.assert_not_called()

    @patch("ft.app.write_state")
    @patch("ft.app.get_state")
    def test_remove_model_success(self, get_state, write_state):
        models: List[ModelMetadata] = [
            ModelMetadata(
                id="model1"
            ),
            ModelMetadata(
                id="model2"
            )
        ]
        get_state.return_value = AppState(
            models=models
        )
        app = generate_test_app()
        app.remove_model("model1")
        write_state.assert_called_with(AppState(
            models=[models[1]]
        ))

    @patch("ft.app.write_state")
    @patch("ft.app.get_state")
    def test_remove_model_no_models(self, get_state, write_state):
        get_state.return_value = AppState()
        app = generate_test_app()
        app.remove_model("model1")
        write_state.assert_called_with(AppState())

    @patch("ft.app.write_state")
    @patch("ft.app.get_state")
    def test_export_model_no_response(self, get_state, write_state):

        # Make a mock model manager
        class MockModelManagerNone(MockModelsManager):
            def export_model(self, request: ExportModelRequest) -> ExportModelResponse:
                return ExportModelResponse()
        app = generate_test_app(models=MockModelManagerNone())
        response: ExportModelResponse = app.export_model(ExportModelRequest(
            type=ModelType.MODEL_TYPE_MODEL_REGISTRY,
            model_id=str(uuid4())
        ))
        assert response.model == ModelMetadata()
        get_state.assert_not_called()
        write_state.assert_not_called()

    @patch("ft.app.write_state")
    @patch("ft.app.get_state")
    def test_export_model_not_model_registry(self, get_state, write_state):
        """
        This test ensures that we are only "adding" exported models back into the
        namespace for model registry models. In reality, this should only be
        tied to a request type object that specifies if we should auto-add the
        model. Similar to auto-adding adapters.
        """

        # Make a mock model manager
        class MockModelManagerNone(MockModelsManager):
            def export_model(self, request: ExportModelRequest) -> ExportModelResponse:
                return ExportModelResponse()
        app = generate_test_app(models=MockModelManagerNone())
        response: ExportModelResponse = app.export_model(ExportModelRequest(
            type=ModelType.MODEL_TYPE_HUGGINGFACE,
            model_id=str(uuid4())
        ))
        assert response.model == ModelMetadata()
        get_state.assert_not_called()
        write_state.assert_not_called()

    @patch("ft.app.write_state")
    @patch("ft.app.get_state")
    def test_export_model_model_registry(self, get_state, write_state):

        model_uuid = str(uuid4())
        cml_model_uuid = str(uuid4())

        # Make a mock model manager
        class MockModelManagerExportWithId(MockModelsManager):
            def export_model(self, request: ExportModelRequest) -> ExportModelResponse:
                return ExportModelResponse(
                    model=ModelMetadata(
                        type=ModelType.MODEL_TYPE_MODEL_REGISTRY,
                        id=model_uuid,
                        registered_model=RegisteredModelMetadata(
                            cml_registered_model_id=cml_model_uuid
                        )
                    )
                )

        get_state.return_value = AppState()

        app = generate_test_app(models=MockModelManagerExportWithId())

        response: ExportModelResponse = app.export_model(ExportModelRequest(
            type=ModelType.MODEL_TYPE_MODEL_REGISTRY,
            model_id=model_uuid
        ))

        assert response.model is not None
        get_state.assert_called_once()
        write_state.assert_called_once()
        write_state.assert_called_with(AppState(
            models=[
                ModelMetadata(
                    type=ModelType.MODEL_TYPE_MODEL_REGISTRY,
                    id=model_uuid,
                    registered_model=RegisteredModelMetadata(
                        cml_registered_model_id=cml_model_uuid
                    )
                )
            ]
        ))
