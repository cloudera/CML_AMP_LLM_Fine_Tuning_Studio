import pytest
from unittest.mock import patch, MagicMock

from sqlalchemy.exc import NoResultFound

import os

from ft.jobs import (
    list_fine_tuning_jobs,
    get_fine_tuning_job,
    remove_fine_tuning_job,
    start_fine_tuning_job,
    _build_argument_list
)
from ft.api import *

from ft.db.dao import FineTuningStudioDao
from ft.db.model import FineTuningJob, Config

from cmlapi import CMLServiceApi

from typing import List


class MockCMLCreatedJob:
    def __init__(self, id: str, script: str = None, runtime_identifier: str = None):
        self.id = id
        self.script = script
        self.runtime_identifier = runtime_identifier


class MockCMLJobRun:
    pass


class MockCMLListJobsResponse:
    def __init__(self, jobs: List[MockCMLCreatedJob] = []):
        self.jobs = jobs


def test_list_fine_tuning_jobs():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(FineTuningJob(id="f1"))
        session.add(FineTuningJob(id="f2", num_epochs=2))

    res: List[FineTuningJobMetadata] = list_fine_tuning_jobs(ListFineTuningJobsRequest(), dao=test_dao).fine_tuning_jobs
    assert len(res) == 2
    assert res[0].id == "f1"
    assert res[1].id == "f2"
    assert res[1].num_epochs == 2


def test_get_fine_tuning_job_happy():

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(FineTuningJob(id="f1"))
        session.add(FineTuningJob(id="f2"))

    res: GetFineTuningJobResponse = get_fine_tuning_job(GetFineTuningJobRequest(id="f2"), dao=test_dao)
    ftjob: FineTuningJobMetadata = res.fine_tuning_job
    assert ftjob.id == "f2"
    assert 'framework_type' not in [x[0].name for x in ftjob.ListFields()]


def test_get_fine_tuning_job_missing():

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with pytest.raises(NoResultFound):
        res = get_fine_tuning_job(GetDatasetRequest(id="f1"), dao=test_dao)


def test_remove_fine_tuning_job_happy():

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(FineTuningJob(id="f1"))
        session.add(FineTuningJob(id="f2"))

    res = remove_fine_tuning_job(RemoveFineTuningJobRequest(id="f1"), dao=test_dao)

    with test_dao.get_session() as session:
        assert len(list(session.query(FineTuningJob).all())) == 1


@patch("ft.jobs.uuid4")
def test_start_fine_tuning_job_happy(uuid4):

    os.environ["CDSW_PROJECT_ID"] = "project_id"
    os.environ["HUGGINGFACE_ACCESS_TOKEN"] = "hf_tok"

    uuid4.return_value = "ftj1"

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    mock_cml = MagicMock(spec=CMLServiceApi)
    mock_cml.list_jobs.return_value = MockCMLListJobsResponse([MockCMLCreatedJob(id="j0")])
    mock_cml.get_job.return_value = MockCMLCreatedJob(id="j0", script="template/script.py", runtime_identifier="rtid")
    mock_cml.create_job.return_value = MockCMLCreatedJob(id="created_job_0")

    res: StartFineTuningJobResponse = start_fine_tuning_job(StartFineTuningJobRequest(), cml=mock_cml, dao=test_dao)
    ftjob: FineTuningJobMetadata = res.fine_tuning_job

    assert ftjob.id == "ftj1"
    assert ftjob.cml_job_id == "created_job_0"
    assert ftjob.framework_type == FineTuningFrameworkType.LEGACY

    with test_dao.get_session() as session:
        assert len(session.query(FineTuningJob).all()) == 1


@patch("ft.jobs.uuid4")
def test_start_fine_tuning_job_custom_config(uuid4):

    os.environ["CDSW_PROJECT_ID"] = "project_id"
    os.environ["HUGGINGFACE_ACCESS_TOKEN"] = "hf_tok"

    uuid4.return_value = "usrcfg1"

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    mock_cml = MagicMock(spec=CMLServiceApi)
    mock_cml.list_jobs.return_value = MockCMLListJobsResponse([MockCMLCreatedJob(id="j0")])
    mock_cml.get_job.return_value = MockCMLCreatedJob(id="j0", script="template/script.py", runtime_identifier="rtid")
    mock_cml.create_job.return_value = MockCMLCreatedJob(id="created_job_0")

    res: StartFineTuningJobResponse = start_fine_tuning_job(StartFineTuningJobRequest(
        user_script="test/script.py",
        user_config='{    \n\n"test": \n"config"}\n\n\n'
    ), cml=mock_cml, dao=test_dao)
    ftjob: FineTuningJobMetadata = res.fine_tuning_job

    assert ftjob.user_config_id == "usrcfg1"
    assert ftjob.user_script == "test/script.py"

    with test_dao.get_session() as session:
        # Added fine tuning job
        assert len(session.query(FineTuningJob).all()) == 1

        # Added new config
        assert len(session.query(Config).all()) == 1
        assert session.get(Config, "usrcfg1").config == '{"test": "config"}'


@patch("os.getenv")
def test_build_argument_list_minimal_request(mock_getenv):
    # Mock environment variables to return None for HUGGINGFACE_ACCESS_TOKEN
    mock_getenv.side_effect = lambda key: None if key == "HUGGINGFACE_ACCESS_TOKEN" else "some_other_value"

    request = StartFineTuningJobRequest(
        base_model_id="base_model_1",
        dataset_id="dataset_1",
        output_dir="/output/dir",
        adapter_name="adapter_1",
        framework_type="LEGACY"
    )
    job_id = "test_job_id"

    arg_list = _build_argument_list(request, job_id)

    expected_arg_list = [
        "--base_model_id", "base_model_1",
        "--dataset_id", "dataset_1",
        "--experimentid", "test_job_id",
        "--out_dir", "/output/dir/test_job_id",
        "--train_out_dir", "outputs/test_job_id",
        "--adapter_name", "adapter_1",
        "--finetuning_framework_type", "LEGACY"
    ]

    assert set(arg_list) == set(expected_arg_list)
    assert "--hf_token" not in arg_list  # Ensure --hf_token is not in the minimal request


@patch("os.getenv")
def test_build_argument_list_with_hf_token(mock_getenv):
    mock_getenv.side_effect = lambda key: "hf_token_value" if key == "HUGGINGFACE_ACCESS_TOKEN" else None

    request = StartFineTuningJobRequest(
        base_model_id="base_model_1",
        dataset_id="dataset_1",
        output_dir="/output/dir",
        adapter_name="adapter_1",
        framework_type="LEGACY"
    )
    job_id = "test_job_id"

    arg_list = _build_argument_list(request, job_id)

    expected_arg_list = [
        "--base_model_id", "base_model_1",
        "--dataset_id", "dataset_1",
        "--experimentid", "test_job_id",
        "--out_dir", "/output/dir/test_job_id",
        "--train_out_dir", "outputs/test_job_id",
        "--adapter_name", "adapter_1",
        "--finetuning_framework_type", "LEGACY",
        "--hf_token", "hf_token_value"
    ]

    assert set(arg_list) == set(expected_arg_list)


@patch("os.getenv")
def test_build_argument_list_with_auto_add_adapter(mock_getenv):
    # Mock environment variables to return None for HUGGINGFACE_ACCESS_TOKEN
    mock_getenv.side_effect = lambda key: None if key == "HUGGINGFACE_ACCESS_TOKEN" else "some_other_value"

    request = StartFineTuningJobRequest(
        base_model_id="base_model_1",
        dataset_id="dataset_1",
        output_dir="/output/dir",
        adapter_name="adapter_1",
        auto_add_adapter=True,
        framework_type="AXOLOTL"
    )
    job_id = "test_job_id"

    arg_list = _build_argument_list(request, job_id)

    expected_arg_list = [
        "--base_model_id", "base_model_1",
        "--dataset_id", "dataset_1",
        "--experimentid", "test_job_id",
        "--out_dir", "/output/dir/test_job_id",
        "--train_out_dir", "outputs/test_job_id",
        "--adapter_name", "adapter_1",
        "--finetuning_framework_type", "AXOLOTL",
        "--auto_add_adapter"
    ]

    assert set(arg_list).issuperset(set(expected_arg_list))
    assert "--hf_token" not in arg_list  # Ensure --hf_token is not in this case unless explicitly needed
