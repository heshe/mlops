import torch.nn.functional as F
from torch import nn


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(16384, 256)
        self.fc2 = nn.Linear(256, 10)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # make sure input tensor is flattened
        # x = x.view(x.shape[0], -1)

        if x.ndim != 4:
            raise ValueError("Expected input to a 4D tensor")
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError(
                f"Expected each sample to have shape [1, 28, 28], but had {x.shape}"
            )

        # Now with dropout
        x = self.dropout(F.relu(self.conv1(x)))
        x = self.dropout(F.relu(self.conv2(x)))
        x = self.dropout(F.relu(self.conv3(x)))

        # Flatten
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))

        # output so no dropout here
        x = F.log_softmax(self.fc2(x), dim=1)

        return x
