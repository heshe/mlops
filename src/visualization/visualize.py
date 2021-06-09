import argparse
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from src.data.make_dataset import load_mnist
from src.models.model import Classifier
from torch import nn

palette = sns.color_palette("bright", 10)


def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default="models/model29.pth")
    parser.add_argument("--input_images", default="")
    args = parser.parse_args()

    # load model
    model = Classifier()
    state_dict = torch.load("models/model29.pth")
    model.load_state_dict(state_dict)

    # set hook:
    activation = {}
    model.fc1.register_forward_hook(get_activation("fc1", activation))

    # Evaluation
    if args.input_images:
        # Implement loading of image
        print("Not supported")
    else:
        _, test_set = load_mnist()
    testloader = torch.utils.data.DataLoader(test_set, batch_size=256,
                                             shuffle=True)

    all_labels = []
    all_outputs = []

    with torch.no_grad():
        model.eval()
        res = torch.zeros(0)
        for images, labels in testloader:

            model(images)
            outputs = activation["fc1"]

            all_labels += labels.tolist()
            all_outputs += outputs.tolist()

    mnist_embedded = TSNE(
        n_components=2,
    ).fit_transform(all_outputs)

    plt.figure(figsize=(8, 5))
    sns_plot = sns.scatterplot(
        mnist_embedded[:, 0],
        mnist_embedded[:, 1],
        hue=all_labels,
        palette=palette,
        legend="full",
    )
    plt.title("Mnist T-SNE plot")
    plt.savefig("reports/figures/tsne_plot.png")
    # plt.show()
