import torch.nn as nn


class LeNet(nn.Module):

    def __init__(self, input_size, in_channels=1, num_classes=10):
        super(LeNet, self).__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.num_classes = num_classes

        def conv2d_size_out(size):
            return (size - 5 + 2 * 0) // 1 + 1

        def maxpool2d_size_out(size):
            return (size - 2) // 2 + 1

        conv1_output_size = conv2d_size_out(input_size)
        pool1_output_size = maxpool2d_size_out(conv1_output_size)
        conv2_output_size = conv2d_size_out(pool1_output_size)
        pool2_output_size = maxpool2d_size_out(conv2_output_size)
        flattened_size = 16 * pool2_output_size * pool2_output_size

        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
