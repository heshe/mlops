# -*- coding: utf-8 -*-
import logging
from pathlib import Path

from torchvision import datasets, transforms


def load_mnist():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Define a transform to normalize the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    project_dir = Path(__file__).resolve().parents[2]

    # Download and load the training data
    trainset = datasets.MNIST(
        str(project_dir) + "/data/MNIST_train/",
        download=True,
        train=True,
        transform=transform,
    )

    # Download and load the test data
    testset = datasets.MNIST(
        str(project_dir) + "/data/MNIST_test/",
        download=True,
        train=False,
        transform=transform,
    )

    return trainset, testset


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files

    # # find .env automagically by walking up directories until it's found, the
    # # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    load_mnist()
