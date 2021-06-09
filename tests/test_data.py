from src.data.make_dataset import load_mnist
import torch


def test_data_loading():
    trainset, testset = load_mnist()

    assert len(trainset) == 60000
    assert len(testset) == 10000
    assert trainset[0][0].shape == torch.Size([1, 28, 28])
    assert len(trainset.classes) == 10

    # assert that each datapoint has shape [1,28,28] or [728] depending on how
    # you choose to format assert that all labels are represented
