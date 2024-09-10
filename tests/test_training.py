import unittest
import datasets

from ft.training.utils import sample_and_split_dataset


class TestSampleAndSplitDataset(unittest.TestCase):

    def setUp(self):
        ds: datasets.Dataset = datasets.Dataset.from_dict({
            "feature1": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "feature2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        self.ds_train = ds
        self.ds_test = ds

    def test_sample_and_split_dataset_not_dict(self):
        """
        This method requires a DatasetDict type as an input.
        """

        with self.assertRaises(AssertionError):
            sample_and_split_dataset(self.ds_train)

    def test_ample_and_split_dataset_no_train_split(self):
        """
        This method requires a 'train' split to be available.
        """

        ds_dict = datasets.DatasetDict({'test': self.ds_test})
        with self.assertRaises(AssertionError):
            sample_and_split_dataset(ds_dict)

    def test_sample_and_split_dataset_deterministic(self):
        """
        Ensures that all randomness is taken out of dataset splitting.
        """

        ds_dict: datasets.DatasetDict = datasets.DatasetDict({'train': self.ds_train})
        ds_train_1, _ = sample_and_split_dataset(ds_dict, train_fraction=0.5, train_test_split=0.5)
        ds_train_2, _ = sample_and_split_dataset(ds_dict, train_fraction=0.5, train_test_split=0.5)
        assert ds_train_1.to_dict() == ds_train_2.to_dict()

    def test_sample_and_split_shuffled_train(self):
        """
        Ensures that a train split is only shuffled if splitting.
        """

        ds_dict: datasets.DatasetDict = datasets.DatasetDict({'train': self.ds_train, 'test': self.ds_test})
        ds_train, _ = sample_and_split_dataset(ds_dict, train_fraction=1.0, train_test_split=0.5)
        assert ds_train.to_dict() == self.ds_train.to_dict()

    def test_sample_and_split_dataset_existing_test_split(self):
        """
        Ensures that if there is a test or eval split, that this split is automatically returned
        as the test split without any dataset splitting.
        """

        ds_dict: datasets.DatasetDict = datasets.DatasetDict({
            'train': self.ds_train,
            'test': self.ds_test
        })
        ds_train, ds_test = sample_and_split_dataset(
            ds_dict,
            train_fraction=0.5,
            train_test_split=0.5)
        assert len(ds_train) == int(0.5 * len(self.ds_train))
        assert ds_test.to_dict() == self.ds_test.to_dict()
