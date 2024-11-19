import pytest
from unittest.mock import patch, mock_open
from sqlalchemy.exc import NoResultFound
import json
import pandas as pd

from ft.evaluation import (
    list_evaluation_jobs,
    get_evaluation_job,
    start_evaluation_job,
    remove_evaluation_job
)

from ft.api import *
from ft.db.dao import FineTuningStudioDao
from ft.db.model import EvaluationJob, Dataset
from ft.eval.mlflow_driver import table_fetcher


class MockCMLCreatedJob:
    def __init__(self, id: str, script: str = None, runtime_identifier: str = None):
        self.id = id
        self.script = script
        self.runtime_identifier = runtime_identifier


class MockCMLListJobsResponse:
    def __init__(self, jobs=None):
        if jobs is None:
            jobs = []
        self.jobs = jobs

# Test: Listing evaluation jobs


def test_list_evaluation_jobs():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(EvaluationJob(id="job1"))
        session.add(EvaluationJob(id="job2"))
        session.commit()

    res = list_evaluation_jobs(ListEvaluationJobsRequest(), dao=test_dao)
    assert len(res.evaluation_jobs) == 2
    assert res.evaluation_jobs[0].id == "job1"
    assert res.evaluation_jobs[1].id == "job2"


# Test: Getting an evaluation job by ID (happy path)
def test_get_evaluation_job_happy():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(EvaluationJob(id="job1"))
        session.add(EvaluationJob(id="job2"))
        session.commit()

    req = GetEvaluationJobRequest(id="job2")
    res = get_evaluation_job(req, dao=test_dao)
    assert res.evaluation_job.id == "job2"


# Test: Getting an evaluation job by ID (job missing)
def test_get_evaluation_job_missing():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with pytest.raises(NoResultFound):
        get_evaluation_job(GetEvaluationJobRequest(id="job1"), dao=test_dao)


# Test: Starting an evaluation job with missing required fields
def test_start_evaluation_job_missing_required_fields():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    request = StartEvaluationJobRequest(
        model_adapter_combinations=[
            EvaluationJobModelCombination(
                base_model_id="model-id",
                adapter_id="adapter-id"
            )
        ],
        dataset_id="dataset-id",
        prompt_id="",  # Required field is empty
        adapter_bnb_config_id="",  # Required field is empty
        model_bnb_config_id="model_bnb_config_id",
        generation_config_id="generation_config_id",
        cpu=4,
        gpu=1,
        memory=16
    )

    with pytest.raises(ValueError) as excinfo:
        start_evaluation_job(request, dao=test_dao)
    assert "Field 'prompt_id' is required in StartEvaluationJobRequest." in str(excinfo.value)


# Test: Starting an evaluation job when referenced model ID does not exist
@patch("ft.evaluation.uuid4")
def test_start_evaluation_job_missing_references(uuid4_mock):
    uuid4_mock.return_value = "job3"

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    # Dataset existence is validated before model/adapter existence, so we need
    # to populate the dataset
    with test_dao.get_session() as session:
        session.add(Dataset(id="dataset-id"))

    request = StartEvaluationJobRequest(
        model_adapter_combinations=[
            EvaluationJobModelCombination(
                base_model_id="non_existent_model_id",
                adapter_id="adapter-id"
            )
        ],
        dataset_id="dataset-id",
        prompt_id="prompt-id",
        adapter_bnb_config_id="adapter_bnb_config_id",
        model_bnb_config_id="model_bnb_config_id",
        generation_config_id="generation_config_id",
        cpu=4,
        gpu=1,
        memory=16
    )

    with pytest.raises(ValueError) as excinfo:
        start_evaluation_job(request, dao=test_dao)
    assert "Model with ID 'non_existent_model_id' does not exist." in str(excinfo.value)


# Test: Removing an evaluation job (happy path)
def test_remove_evaluation_job_happy():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(EvaluationJob(id="job1"))
        session.add(EvaluationJob(id="job2"))
        session.commit()

    req = RemoveEvaluationJobRequest(id="job1")
    res = remove_evaluation_job(req, dao=test_dao)

    with test_dao.get_session() as session:
        jobs = session.query(EvaluationJob).all()
        assert len(jobs) == 1
        assert jobs[0].id == "job2"

# Test: Fetching table from evaluation Job


@pytest.fixture
def mock_results():
    class MockArtifact:
        def __init__(self, uri):
            self.uri = uri

    class MockResults:
        artifacts = {
            'eval_results_table': MockArtifact('path/to/eval_results_table.json')
        }

    return MockResults()


@pytest.fixture
def mock_json_data():
    return {
        "columns": ["col1", "col2", "col3"],
        "data": [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
    }


def test_table_fetcher_with_mock_results(mock_results, mock_json_data):
    with patch("builtins.open", mock_open(read_data=json.dumps(mock_json_data))):
        df = table_fetcher(mock_results)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert list(df.columns) == ["col1", "col2", "col3"]
    assert df.iloc[0, 0] == 1
    assert df.iloc[1, 1] == 5
    assert df.iloc[2, 2] == 9


def test_table_fetcher_with_invalid_json(mock_results):
    with patch("builtins.open", mock_open(read_data="invalid json")), \
            pytest.raises(json.JSONDecodeError):
        table_fetcher(mock_results)
