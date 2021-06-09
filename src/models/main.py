import argparse
import sys

import matplotlib.pyplot as plt
import torch
import tqdm
from src.data.make_dataset import load_mnist
import torchvision
from torch import nn, optim

# from torch.utils.tensorboard import SummaryWriter
import wandb

from src.models.model import Classifier


class TrainOREvaluate(object):
    """Helper class that will help launch class methods as commands
    from a single script
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>",
        )
        parser.add_argument("command", help="Subcommand to run")
        self.args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, self.args.command):
            print("Unrecognized command")

            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, self.args.command)()

    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--lr", default=0.1)
        parser.add_argument("--n_epochs", default=30, type=int)
        parser.add_argument("--run_name", default="new")
        parser.add_argument("--use_wandb", default=True, type=bool)
        parser.add_argument("--plot_results", default=True, type=bool)

        # add any additional argument that you want
        self.args = parser.parse_args(sys.argv[2:])
        print(self.args)

        if self.args.use_wandb:
            wandb.init(config=self.args)

        # ____ Setup model, loss and optimizer _____
        model = Classifier()
        if self.args.use_wandb:
            wandb.watch(model, log_freq=100)
        optimizer = optim.Adam(model.parameters(), lr=0.003)
        criterion = nn.NLLLoss()

        # ____ Loop Variables ____
        n_epochs = self.args.n_epochs
        max_steps = 10
        train_losses, test_losses, epochs = [], [], []
        train_set, test_set = load_mnist()
        trainloader = torch.utils.data.DataLoader(
            train_set, batch_size=256, shuffle=True
        )
        testloader = torch.utils.data.DataLoader(test_set, batch_size=256,
                                                 shuffle=True)

        # ____Training loop  _____
        for e in range(n_epochs):
            train_loss = 0
            test_loss = 0
            steps = 0
            for images, labels in tqdm.tqdm(trainloader, total=max_steps):
                if steps > max_steps:
                    break

                # TRAIN
                model.train()

                optimizer.zero_grad()

                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                steps += 1

            # VALIDATION
            with torch.no_grad():
                model.eval()
                res = torch.zeros(0)
                for images, labels in testloader:
                    # Get the val loss
                    log_ps = model(images)
                    loss = criterion(log_ps, labels)
                    test_loss += loss.item()

                    # Get the class probabilities
                    ps = torch.exp(log_ps)
                    _, top_class = ps.topk(1, dim=1)

                    equals = top_class == labels.view(*top_class.shape)
                    res = torch.cat((res, equals), dim=0)
                accuracy = torch.mean(res.type(torch.FloatTensor))

            # Wandb
            if self.args.use_wandb:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "test_loss": test_loss,
                        "accuracy": accuracy,
                    }
                )
                wandb.log({"examples": [wandb.Image(i) for i in images]})

            # Save current model
            if e % 5 == 0:
                torch.save(
                    model.state_dict(), 
                    f"models/{self.args.run_name}_model{e}.pth"
                )

            # Sum up epoch
            epochs += [e]
            train_losses += [train_loss]
            test_losses += [test_loss]
            print(f"Epoch {e}:   acc={round(accuracy.item()*100, 4)}%")
            savedir = f"reports/figures/loss_curve_{self.args.run_name}.pdf"
            if self.args.plot_results:
                plt.plot(epochs, train_losses, label="Train")
                plt.plot(epochs, test_losses, label="Test")
                plt.legend()
                plt.savefig(savedir)
                plt.close()

        return train_losses, test_losses

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--load_model_from", default="")
        # add any additional argument that you want
        self.args = parser.parse_args(sys.argv[2:])
        print(self.args)

        # load model
        model = Classifier()
        if self.args.load_model_from:
            state_dict = torch.load(self.args.load_model_from)
        model.load_state_dict(state_dict)

        # Evaluation
        _, test_set = load_mnist()
        testloader = torch.utils.data.DataLoader(test_set, batch_size=256,
                                                 shuffle=True)
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

        print("Accuracy is: ", accuracy)


if __name__ == "__main__":
    TrainOREvaluate()
