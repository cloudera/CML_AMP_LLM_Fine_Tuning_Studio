from typing import List
from unittest.mock import patch
from uuid import uuid4
import pytest


from ft.managers.models import ModelsManagerBase, ModelsManagerSimple
from ft.managers.datasets import DatasetsManagerBase, DatasetsManagerSimple
from ft.managers.jobs import FineTuningJobsManagerBase, FineTuningJobsManagerSimple

from ft.app import (
    FineTuningApp,
    FineTuningAppProps
)

from ft.model import (
    ExportModelRequest,
    ExportModelResponse,
    ExportType,
    ImportModelRequest,
    ImportModelResponse,
    ModelMetadata,
    RegisteredModelMetadata
)

from ft.state import AppState


class MockModelManager(ModelsManagerBase):
    def export_model(self, request: ExportModelRequest) -> ExportModelResponse:
        pass

    def import_model(self, request: ImportModelRequest) -> ImportModelResponse:
        pass

    def list_models(self) -> List[ModelMetadata]:
        pass


class TestAppModels():

    def mock_app_state_empty(self):
        return AppState(
            datasets=[],
            models=[],
            jobs=[],
            prompts=[],
            adapters=[],
            registered_models=[]
        )

    @patch("ft.app.update_state")
    @patch("ft.app.get_state")
    def test_export_model_no_response(self, get_state, update_state):

        # Make a mock model manager
        class MockModelManagerNone(MockModelManager):
            def export_model(self, request: ExportModelRequest) -> ExportModelResponse:
                return ExportModelResponse()

        app = FineTuningApp(FineTuningAppProps(
            DatasetsManagerSimple(),
            MockModelManagerNone(),
            FineTuningJobsManagerSimple()
        ))

        response: ExportModelResponse = app.export_model(ExportModelRequest(
            type=ExportType.MODEL_REGISTRY,
            model_id=str(uuid4())
        ))

        assert response.registered_model is None
        get_state.assert_not_called()
        update_state.assert_not_called()

    @patch("ft.app.update_state")
    @patch("ft.app.get_state")
    def test_export_model_not_model_registry(self, get_state, update_state):

        # Make a mock model manager
        class MockModelManagerExport(MockModelManager):
            def export_model(self, request: ExportModelRequest) -> ExportModelResponse:
                return ExportModelResponse(
                    registered_model=RegisteredModelMetadata(
                        model_id=str(uuid4()),
                        cml_model_id=str(uuid4())
                    )
                )

        app = FineTuningApp(FineTuningAppProps(
            DatasetsManagerSimple(),
            MockModelManagerExport(),
            FineTuningJobsManagerSimple()
        ))

        response: ExportModelResponse = app.export_model(ExportModelRequest(
            type=ExportType.ONNX,
            model_id=str(uuid4())
        ))

        assert response.registered_model is not None
        get_state.assert_not_called()
        update_state.assert_not_called()

    @patch("ft.app.update_state")
    @patch("ft.app.get_state")
    def test_export_model_model_registry(self, get_state, update_state):

        model_uuid = str(uuid4())
        cml_model_uuid = str(uuid4())

        # Make a mock model manager
        class MockModelManagerExportWithId(MockModelManager):
            def export_model(self, request: ExportModelRequest) -> ExportModelResponse:
                return ExportModelResponse(
                    registered_model=RegisteredModelMetadata(
                        model_id=request.model_id,
                        cml_model_id=cml_model_uuid
                    )
                )

        get_state.return_value = self.mock_app_state_empty()

        app = FineTuningApp(FineTuningAppProps(
            DatasetsManagerSimple(),
            MockModelManagerExportWithId(),
            FineTuningJobsManagerSimple()
        ))

        response: ExportModelResponse = app.export_model(ExportModelRequest(
            type=ExportType.MODEL_REGISTRY,
            model_id=model_uuid
        ))

        assert response.registered_model is not None
        get_state.assert_called_once()
        update_state.assert_called_once()
        update_state.assert_called_with({
            "registered_models": [
                RegisteredModelMetadata(
                    model_id=model_uuid,
                    cml_model_id=cml_model_uuid
                )
            ]
        })
