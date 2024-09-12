import unittest
from unittest.mock import patch
from ft.fine_tune_axolotl import AxolotlFineTuner

# Assuming the AxolotlFineTuner class is defined in the same module or is imported


class TestAxolotlFineTuner(unittest.TestCase):

    @patch('mlflow.set_experiment')
    def test_init_with_valid_uuid(self, mock_set_experiment):
        ft_job_uuid = "12345"
        tuner = AxolotlFineTuner(ft_job_uuid)
        mock_set_experiment.assert_called_once_with(ft_job_uuid)

    def test_init_without_uuid(self):
        with self.assertRaises(ValueError):
            AxolotlFineTuner()
