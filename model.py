import torch
import torch.nn as nn
import torch.nn.functional as F


class OutputPredictor(nn.Module):
    def __init__(self):
        super(OutputPredictor, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.fc1_2d = nn.Linear(16 * 100 * 100, 256)

        self.fc1_1d_real = nn.Linear(20, 128)
        self.fc1_1d_imag = nn.Linear(20, 128)

        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 40)  # just real concat with imag

    def forward(self, x2d, x1d_real, x1d_imag):
        # Assume x2d is (batch_size, 100, 100) and needs an extra channel dimension
        x2d = x2d.unsqueeze(1)  # Now x2d is (batch_size, 1, 100, 100)
        x2d = F.relu(self.conv1(x2d))
        x2d = self.bn1(x2d)
        x2d = x2d.view(x2d.size(0), -1)  # Flatten
        x2d = F.relu(self.fc1_2d(x2d))

        # Process the 1D array
        x1d_real = F.relu(self.fc1_1d_real(x1d_real))
        x1d_imag = F.relu(self.fc1_1d_imag(x1d_imag))

        # Concatenate along the feature dimension
        x = torch.cat((x2d, x1d_real, x1d_imag), dim=1)
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = self.fc5(x)
        return x
