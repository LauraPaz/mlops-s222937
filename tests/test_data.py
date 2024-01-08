import os

import pytest

from src.data.make_dataset import mnist


@pytest.mark.skipif(not os.path.exists('data/raw'), reason="Data files not found")
def test_data():
    N_train = 50000
    N_test = 5000

    train_set, test_set = mnist()

    assert len(train_set) == N_train, "Train dataset did not have the correct number of samples"
    assert len(test_set) == N_test, "Test dataset did not have the correct number of samples"

    # assert that each datapoint has shape [1,28,28]
    assert train_set[0][0].shape == (1,28,28)

    # assert that all labels (numbers from 0 to 9) are represented
    assert set(train_set.tensors[1].numpy()) == set(range(10))
    assert set(test_set.tensors[1].numpy()) == set(range(10))