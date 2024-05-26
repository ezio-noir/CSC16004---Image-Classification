import torch

import torch.nn as nn


class Ann(nn.Module):

    def __init__(self, input_size, num_classes=10):
        super(Ann, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_layers = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, self.num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.hidden_layers(x)
        x = self.classifier(x)
        return x
