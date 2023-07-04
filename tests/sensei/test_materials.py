import pytest
import torch
from torch.utils.data import TensorDataset

from zenkai.sensei import materials


class TestDLMaterial:

    def test_loops_over_dataset_with_dataloader(self):

        target_x = torch.rand(4, 2)
        target_t = torch.rand(4)
        dataset = TensorDataset(target_x, target_t)
        dl_material = materials.DLMaterial.load(
            dataset, 2, False
        )
        xs = []
        ts = []
        for x, t in dl_material:
            xs.append(x)
            ts.append(t)

        assert (torch.cat(xs) == target_x).all()
        assert (torch.cat(ts) == target_t).all()


class TestSplitDataset:

    def test_split_dataset_splits_into_training_and_test(self):

        target_x = torch.rand(4, 2)
        target_t = torch.rand(4)
        dataset = TensorDataset(target_x, target_t)
        training, test = materials.split_dataset(dataset, 0.5, True)
        assert len(training) == 2
        assert len(test) == 2

    def test_split_dataset_splits_into_training_and_test_and_validation(self):

        target_x = torch.rand(4, 2)
        target_t = torch.rand(4)
        dataset = TensorDataset(target_x, target_t)
        training, test, validation = materials.split_dataset(dataset, [0.25, 0.5], True)
        assert len(training) == 1
        assert len(test) == 1
        assert len(validation) == 2

    def test_split_dataset_splits_into_training_and_test_and_validation_with_indices(self):

        target_x = torch.rand(4, 2)
        target_t = torch.rand(4)
        dataset = TensorDataset(target_x, target_t)
        training, test, validation = materials.split_dataset(dataset, [1, 2], True)
        assert len(training) == 1
        assert len(test) == 1
        assert len(validation) == 2

    def test_split_dataset_raises_error_if_invalid_split_locations(self):

        target_x = torch.rand(4, 2)
        target_t = torch.rand(4)
        dataset = TensorDataset(target_x, target_t)
        with pytest.raises(ValueError): 
            materials.split_dataset(dataset, [0.0, 0.0], True)

    def test_split_dataset_raises_error_if_invalid_split_locations_with_indices(self):

        target_x = torch.rand(4, 2)
        target_t = torch.rand(4)
        dataset = TensorDataset(target_x, target_t)
        with pytest.raises(ValueError): 
            materials.split_dataset(dataset, [1, 4], True)

