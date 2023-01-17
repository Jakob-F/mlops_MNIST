import pytest
import torch
from tests import _PATH_DATA
import os

def test_train_data():
    train_data_path = os.path.join(_PATH_DATA, "train.pt")  # root of data
    dataset = torch.load(train_data_path)


    # assert size of each datapoint
    assert len(dataset) == 50000


# def test_dummy():
#     assert True

# def test_dummy_cache():
#     assert True
