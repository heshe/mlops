import argparse
import sys

import matplotlib.pyplot as plt
import torch
import tqdm
from src.data.make_dataset import load_mnist
import torchvision
from torch import nn, optim
#from torch.utils.tensorboard import SummaryWriter
import wandb
from src.models.model import Classifier
import argparse


class Trainer():
    def __init__(self):
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--lr", default=0.1, type=float)
        parser.add_argument("--n_epochs", default=30, type=int)
        parser.add_argument("--run_name", default="new")
        parser.add_argument("--use_wandb", default=True, type=bool)
        parser.add_argument("--plot_results", default=True, type=bool)

        # add any additional argument that you want
        self.args = parser.parse_args(sys.argv[2:])
        print(sys.argv)

    def train(self):
        
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
        testloader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=True)

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
                wandb.log({"train_loss": train_loss, "test_loss": test_loss,
                        "accuracy": accuracy})
                wandb.log({"examples": [wandb.Image(i) for i in images]})

            # Save current model
            if e % 5 == 0:
                torch.save(model.state_dict(), f"models/{self.args.run_name}_model{e}.pth")

            # Sum up epoch
            epochs += [e]
            train_losses += [train_loss]
            test_losses += [test_loss]
            print(f"Epoch {e}:   acc={round(accuracy.item()*100, 4)}%")
            if self.args.plot_results:
                plt.plot(epochs, train_losses, label="Train")
                plt.plot(epochs, test_losses, label="Test")
                plt.legend()
                plt.savefig(f"reports/figures/loss_curve_{self.args.run_name}.pdf")
                plt.close()

        return train_losses, test_losses


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
