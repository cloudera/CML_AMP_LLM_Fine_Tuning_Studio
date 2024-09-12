import pytest
from unittest.mock import patch, MagicMock

from sqlalchemy.exc import NoResultFound

import os
import yaml
import json

from ft.jobs import (
    list_fine_tuning_jobs,
    get_fine_tuning_job,
    remove_fine_tuning_job,
    start_fine_tuning_job,
    _build_argument_list,
    _validate_fine_tuning_request,
    _add_prompt_for_dataset
)
from ft.api import *

from ft.db.dao import FineTuningStudioDao
from ft.db.model import FineTuningJob, Config, Dataset, Prompt, Model

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
        session.add(FineTuningJob(id="f1", framework_type=FineTuningFrameworkType.LEGACY))
        session.add(FineTuningJob(id="f2", framework_type=FineTuningFrameworkType.AXOLOTL))

    res: GetFineTuningJobResponse = get_fine_tuning_job(GetFineTuningJobRequest(id="f2"), dao=test_dao)
    ftjob: FineTuningJobMetadata = res.fine_tuning_job
    assert ftjob.id == "f2"
    assert ftjob.framework_type == FineTuningFrameworkType.AXOLOTL


def test_get_fine_tuning_job_missing():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with pytest.raises(NoResultFound):
        get_fine_tuning_job(GetFineTuningJobRequest(id="f1"), dao=test_dao)


def test_remove_fine_tuning_job_happy():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        session.add(FineTuningJob(id="f1"))
        session.add(FineTuningJob(id="f2"))

    remove_fine_tuning_job(RemoveFineTuningJobRequest(id="f1"), dao=test_dao)

    with test_dao.get_session() as session:
        assert len(list(session.query(FineTuningJob).all())) == 1


@patch("ft.jobs.uuid4")
def test_start_fine_tuning_job_happy(uuid4):
    os.environ["CDSW_PROJECT_ID"] = "project_id"
    os.environ["HUGGINGFACE_ACCESS_TOKEN"] = "hf_tok"

    uuid4.return_value = "ftj1"

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    # Add necessary entries to the test DAO for validation
    with test_dao.get_session() as session:
        model = Model(id="model_id", name="Test Model")
        session.add(model)

        dataset = Dataset(id="dataset_id", name="Test Dataset")
        session.add(dataset)

        axolotl_config = Config(id="axolotl_cfg_id", config="test config", description="Axolotl config")
        session.add(axolotl_config)

        training_args_config = Config(id="training_args_cfg_id", config="training arguments config")
        session.add(training_args_config)

        session.commit()

    mock_cml = MagicMock(spec=CMLServiceApi)
    mock_cml.list_jobs.return_value = MockCMLListJobsResponse([MockCMLCreatedJob(id="j0")])
    mock_cml.get_job.return_value = MockCMLCreatedJob(id="j0", script="template/script.py", runtime_identifier="rtid")
    mock_cml.create_job.return_value = MockCMLCreatedJob(id="created_job_0")

    res: StartFineTuningJobResponse = start_fine_tuning_job(StartFineTuningJobRequest(
        base_model_id="model_id",
        dataset_id="dataset_id",
        adapter_name="test-adapter",
        cpu=1,  # Ensure valid CPU allocation
        gpu=1,
        memory=1024,
        num_workers=1,  # Ensure valid number of workers
        num_epochs=1,  # Ensure valid number of epochs
        learning_rate=0.001,  # Ensure valid learning rate
        dataset_fraction=0.9,
        train_test_split=0.9,
        training_arguments_config_id="training_args_cfg_id"
    ), cml=mock_cml, dao=test_dao)
    ftjob: FineTuningJobMetadata = res.fine_tuning_job

    assert ftjob.id == "ftj1"
    assert ftjob.cml_job_id == "created_job_0"

    with test_dao.get_session() as session:
        assert len(session.query(FineTuningJob).all()) == 1


@patch("ft.jobs.uuid4")
def test_start_fine_tuning_job_custom_config(uuid4):
    os.environ["CDSW_PROJECT_ID"] = "project_id"
    os.environ["HUGGINGFACE_ACCESS_TOKEN"] = "hf_tok"

    uuid4.return_value = "usrcfg1"

    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    # Add necessary entries to the test DAO for validation
    with test_dao.get_session() as session:
        model = Model(id="model_id", name="Test Model")
        session.add(model)

        dataset = Dataset(id="dataset_id", name="Test Dataset")
        session.add(dataset)

        axolotl_config = Config(id="axolotl_cfg_id", config="test config", description="Axolotl config")
        session.add(axolotl_config)

        training_args_config = Config(id="training_args_cfg_id", config="training arguments config")
        session.add(training_args_config)

        session.commit()

    mock_cml = MagicMock(spec=CMLServiceApi)
    mock_cml.list_jobs.return_value = MockCMLListJobsResponse([MockCMLCreatedJob(id="j0")])
    mock_cml.get_job.return_value = MockCMLCreatedJob(id="j0", script="template/script.py", runtime_identifier="rtid")
    mock_cml.create_job.return_value = MockCMLCreatedJob(id="created_job_0")

    res: StartFineTuningJobResponse = start_fine_tuning_job(StartFineTuningJobRequest(
        user_script="test/script.py",
        user_config='{    \n\n"test": \n"config"}\n\n\n',
        base_model_id="model_id",
        dataset_id="dataset_id",
        adapter_name="test-adapter",
        cpu=1,  # Ensure valid CPU allocation
        gpu=1,
        memory=1024,
        num_workers=1,  # Ensure valid number of workers
        num_epochs=1,  # Ensure valid number of epochs
        learning_rate=0.001,  # Ensure valid learning rate
        dataset_fraction=0.9,
        train_test_split=0.9
    ), cml=mock_cml, dao=test_dao)
    ftjob: FineTuningJobMetadata = res.fine_tuning_job

    assert ftjob.user_config_id == "usrcfg1"
    assert ftjob.user_script == "test/script.py"

    with test_dao.get_session() as session:
        # Added fine-tuning job
        assert len(session.query(FineTuningJob).all()) == 1

        # Added new config
        assert len(session.query(Config).all()) == 3  # Includes axolotl_cfg_id and usrcfg1
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
        framework_type="LEGACY",
        cpu=1,  # Ensure valid CPU allocation
        gpu=1,
        memory=1024,
        num_workers=1,  # Ensure valid number of workers
        num_epochs=1,  # Ensure valid number of epochs
        learning_rate=0.001,  # Ensure valid learning rate
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
        framework_type="LEGACY",
        cpu=1,  # Ensure valid CPU allocation
        gpu=1,
        memory=1024,
        num_workers=1,  # Ensure valid number of workers
        num_epochs=1,  # Ensure valid number of epochs
        learning_rate=0.001,  # Ensure valid learning rate
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
        framework_type="AXOLOTL",
        cpu=1,  # Ensure valid CPU allocation
        gpu=1,
        memory=1024,
        num_workers=1,  # Ensure valid number of workers
        num_epochs=1,  # Ensure valid number of epochs
        learning_rate=0.001,  # Ensure valid learning rate
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


# Additional tests

def test_validate_fine_tuning_request_invalid_framework_type():
    request = StartFineTuningJobRequest(
        framework_type="INVALID_FRAMEWORK_TYPE",
        cpu=1,
        gpu=1,
        memory=1024,
        num_workers=1,
        num_epochs=1,
        learning_rate=0.001)
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with pytest.raises(ValueError, match="Invalid framework type provided."):
        _validate_fine_tuning_request(request, test_dao)


def test_validate_fine_tuning_request_invalid_adapter_name():
    request = StartFineTuningJobRequest(
        adapter_name="Invalid Adapter Name",
        cpu=1,
        gpu=1,
        memory=1024,
        num_workers=1,
        num_epochs=1,
        learning_rate=0.001,
        framework_type=FineTuningFrameworkType.LEGACY)
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with pytest.raises(ValueError, match="Adapter Name should only contain alphanumeric characters and hyphens, and no spaces."):
        _validate_fine_tuning_request(request, test_dao)


def test_validate_fine_tuning_request_invalid_output_dir():
    request = StartFineTuningJobRequest(
        output_dir="/invalid/output/dir",
        cpu=1,
        gpu=1,
        memory=1024,
        num_workers=1,
        num_epochs=1,
        learning_rate=0.001,
        framework_type=FineTuningFrameworkType.LEGACY)
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with patch("os.path.isdir", return_value=False):
        with pytest.raises(ValueError, match="Output Location must be a valid folder directory."):
            _validate_fine_tuning_request(request, test_dao)


def test_add_prompt_for_dataset_happy_path():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        dataset_id = "ds1"
        axolotl_config_id = "axolotl_cfg1"
        session.add(Dataset(id=dataset_id, name="Dataset1"))
        session.add(Config(id=axolotl_config_id, config=yaml.dump({"datasets": [{"type": "dataset_type"}]})))
        session.add(Config(id="config1", description="dataset_type", config=json.dumps({"feature1": "value1"})))

    prompt_id = _add_prompt_for_dataset(dataset_id, axolotl_config_id, dao=test_dao)

    with test_dao.get_session() as session:
        prompt = session.query(Prompt).filter_by(id=prompt_id).one()
        assert prompt is not None
        assert prompt.dataset_id == dataset_id
        assert "feature1" in prompt.prompt_template


def test_add_prompt_for_dataset_no_axolotl_config():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        dataset_id = "ds1"
        axolotl_config_id = "axolotl_cfg1"
        session.add(Dataset(id=dataset_id, name="Dataset1"))

    with pytest.raises(ValueError, match=f"No configuration found with id {axolotl_config_id}."):
        _add_prompt_for_dataset(dataset_id, axolotl_config_id, dao=test_dao)


def test_add_prompt_for_dataset_no_dataset_type():
    test_dao = FineTuningStudioDao(engine_url="sqlite:///:memory:", echo=False)

    with test_dao.get_session() as session:
        dataset_id = "ds1"
        axolotl_config_id = "axolotl_cfg1"
        session.add(Dataset(id=dataset_id, name="Dataset1"))
        session.add(Config(id=axolotl_config_id, config=yaml.dump({"datasets": [{}]})))

    with pytest.raises(ValueError, match="Dataset type could not be found in the configuration."):
        _add_prompt_for_dataset(dataset_id, axolotl_config_id, dao=test_dao)
