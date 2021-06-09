import argparse
import sys

import matplotlib.pyplot as plt
import torch
from src.data.make_dataset import load_mnist

from src.models.model import Classifier


def predict(model_path, input_images, shuffle=True):
    # load model
    model = Classifier()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    # Evaluation
    if input_images:
        # Implement loading of image
        print("Not supported")
    else:
        _, test_set = load_mnist()
    testloader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=shuffle)
    with torch.no_grad():
        model.eval()
        res = torch.zeros(0)
        for images, labels in testloader:
            # Get the val loss
            log_ps = model(images)

            # Get the class probabilities
            ps = torch.exp(log_ps)
            _, top_class = ps.topk(1, dim=1)

            equals = top_class == labels.view(*top_class.shape)
            res = torch.cat((res, equals), dim=0)
        accuracy = torch.mean(res.type(torch.FloatTensor))

    print("Accuracy is: ", accuracy.item())

    return accuracy.item(), log_ps, images


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default="models/model29.pth")
    parser.add_argument("--input_images", default="")
    args = parser.parse_args()
    predict(args.model_path, args.input_images)
