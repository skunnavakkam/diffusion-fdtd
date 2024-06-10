import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_criterion(output, target):
    # Ensure output and target are of the same shape
    assert output.shape == target.shape, "Output and target must have the same shape"

    # Split the output and target into two parts
    output_first, output_second = output[:20], output[20:40]
    target_first, target_second = target[:20], target[20:40]

    # Calculate MSE for the first 20 elements
    mse_first = F.mse_loss(output_first, target_first, reduction="mean")

    # Calculate MSE for the second 20 elements
    mse_second = F.mse_loss(output_second, target_second, reduction="mean")

    # Multiply the two MSEs
    loss = mse_first * mse_second
    return loss


class OutputPredictor(nn.Module):
    def __init__(self):
        super(OutputPredictor, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.fc1_2d = nn.Linear(8 * 100 * 100, 512)

        self.fc1_1d_real = nn.Linear(20, 256)
        self.fc1_1d_imag = nn.Linear(20, 256)

        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout = nn.Dropout(0.2)

        self.fc2 = nn.Linear(1024, 384)
        self.fc3 = nn.Linear(384, 128)
        self.fc5 = nn.Linear(128, 40)  # just real concat with imag

    def forward(self, x2d, x1d_real, x1d_imag):
        # Assume x2d is (batch_size, 100, 100) and needs an extra channel dimension
        x2d = x2d.unsqueeze(1)  # Now x2d is (batch_size, 1, 100, 100)
        x2d = F.relu(self.conv1(x2d))
        x2d = x2d.view(x2d.size(0), -1)  # Flatten
        x2d = F.relu(self.fc1_2d(x2d))

        # Process the 1D array
        x1d_real = F.relu(self.fc1_1d_real(x1d_real))
        x1d_imag = F.relu(self.fc1_1d_imag(x1d_imag))

        # Concatenate along the feature dimension
        x = torch.cat((x2d, x1d_real, x1d_imag), dim=1)
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc5(x)
        return x
