import numpy as np
from perlin_numpy import generate_perlin_noise_2d
from scipy import ndimage
import autograd.numpy as npa
import matplotlib as mpl
import matplotlib.pylab as plt
import ceviche
from ceviche import fdfd_ez
from ceviche.modes import insert_mode
import collections
from random import randint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import torch as t
from torch.utils.data import DataLoader, TensorDataset, random_split

# Create a container for our slice coords to be used for sources and probes
# this is going to be super basic, where we take one input and one output to train
Slice = collections.namedtuple("Slice", "x y")
mpl.rcParams["figure.dpi"] = 100

Nx = Ny = 100
Npml = 10
space = 10
wg_width = 12
epsr_init = 12
dev_Nx = dev_Ny = 60
space_slice = 4
omega = 2 * np.pi * 200e12
dl = 40e-9


class OutputPredictor(nn.Module):
    def __init__(self):
        super(OutputPredictor, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1_2d = nn.linear(32 * 100 * 100, 256)

        self.fc1_1d = nn.linear(20, 256)

        self.fc2 = nn.linear(512, 256)
        self.fc3 = nn.linear(256, 128)
        self.fc4 = nn.linear(128, 20)

    def forward(self, x2d, x1d):
        # Assume x2d is (batch_size, 100, 100) and needs an extra channel dimension
        x2d = x2d.unsqueeze(1)  # Now x2d is (batch_size, 1, 100, 100)
        x2d = F.relu(self.conv1(x2d))
        x2d = F.relu(self.conv2(x2d))
        x2d = x2d.view(x2d.size(0), -1)  # Flatten
        x2d = F.relu(self.fc1_2d(x2d))

        # Process the 1D array
        x1d = F.relu(self.fc1_1d(x1d))

        # Concatenate along the feature dimension
        x = torch.cat((x2d, x1d), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def dfs(matrix):
    """
    Determines if there is material connecting the input and output waveguides

    Args:
    numpy.ndarray: A 2D array representing the permittivity of the device.

    Returns:
    bool: True if there is a path connecting the input and output waveguides, False otherwise
    """

    rows, cols = matrix.shape
    start = (0, cols // 2)
    end = (rows - 1, cols // 2)

    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Stack for DFS
    stack = [start]
    visited = set()
    visited.add(start)

    while stack:
        x, y = stack.pop()

        # If we've reached the end point
        if (x, y) == end:
            return True

        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check bounds and if it's a valid move
            if (
                0 <= nx < rows
                and 0 <= ny < cols
                and matrix[nx, ny] == epsr_init
                and (nx, ny) not in visited
            ):
                stack.append((nx, ny))
                visited.add((nx, ny))

    return False


def generate_modes():
    pass


def generate_perlin_device(width, height):
    """
    Returns:
    numpy.ndarray: A 2D array (width, height) representing the device.
    """
    treshold = 0.05
    device = generate_perlin_noise_2d((width, height), (2, 2)) > treshold
    return device


def init_domain(Nx, Ny, Npml, space=10, wg_width=10, space_slice=5):
    """Initializes the domain and design region

    space       : The space between the PML and the structure
    wg_width    : The feed and probe waveguide width
    space_slice : The added space for the probe and source slices

    Returns:

    """

    # Parametrization of the permittivity of the structure
    bg_epsr = np.zeros((Nx, Ny))
    epsr = np.zeros((Nx, Ny))

    # Region within which the permittivity is allowed to change
    design_region = np.zeros((Nx, Ny))

    # Input waveguide
    bg_epsr[
        0 : int(Npml + space), int(Ny / 2 - wg_width / 2) : int(Ny / 2 + wg_width / 2)
    ] = epsr_init

    # Input probe slice
    input_slice = Slice(
        x=np.array(Npml + 1),
        y=np.arange(
            int(Ny / 2 - wg_width / 2 - space_slice),
            int(Ny / 2 + wg_width / 2 + space_slice),
        ),
    )

    # Output waveguide
    bg_epsr[
        int(Nx - Npml - space) : :,
        int(Ny / 2 - wg_width / 2) : int(Ny / 2 + wg_width / 2),
    ] = epsr_init

    # Output probe slice
    output_slice = Slice(
        x=np.array(Nx - Npml - 1),
        y=np.arange(
            int(Ny / 2 - wg_width / 2 - space_slice),
            int(Ny / 2 + wg_width / 2 + space_slice),
        ),
    )

    design_region[
        Npml + space : Nx - Npml - space, Npml + space : Ny - Npml - space
    ] = 1
    epsr[Npml + space : Nx - Npml - space, Npml + space : Ny - Npml - space] = epsr_init

    ret = final_epsr = epsr * design_region + bg_epsr * (design_region == 0).astype(
        float
    )
    # generates the device
    ret[20:80, 20:80] = generate_perlin_device(60, 60) * epsr_init
    # smoothing to remove sharp edges (makes the simulation go weird)
    ret = ndimage.median_filter(ret, size=10)
    final_epsr[20:80, 20:80] = ret[20:80, 20:80]

    return final_epsr, input_slice, output_slice


def mask_combine_epsr(epsr, bg_epsr, design_region):
    """Utility function for combining the design region epsr and the background epsr"""
    return epsr * design_region + bg_epsr * (design_region == 0).astype(float)


def viz_sim(epsr, source, slices=[]):
    """Solve and visualize a simulation with permittivity 'epsr'"""
    simulation = fdfd_ez(omega, dl, epsr, [Npml, Npml])
    Hx, Hy, Ez = simulation.solve(source)
    fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(6, 3))
    ceviche.viz.real(Ez, outline=epsr, ax=ax[0], cbar=False)
    for sl in slices:
        ax[0].plot(sl.x * np.ones(len(sl.y)), sl.y, "b-")
    ceviche.viz.abs(epsr, ax=ax[1], cmap="Greys")
    plt.show()
    return (simulation, ax)


def mode_overlap(E1, E2):
    """Defines an overlap integral between the simulated field and desired field"""
    return npa.abs(npa.sum(npa.conj(E1) * E2))


def generate_new_device():
    """
    Generates and simulates a new, randomly generated device

    Returns:
    numpy.ndarray: A 2D array representing the permittivity of the device. Only takes on values of 1 or 12.
    numpy.ndarray: A 1D array representing the field at the input slice
    numpy.ndarray: A 1D array representing the field at the output slice
    """

    # parameters for the device
    num_input_modes = randint(1, 5)

    epsr = np.zeros((Nx, Ny))
    while not dfs(epsr):
        epsr, input_slice, output_slice = init_domain(
            Nx, Ny, Npml, space=space, wg_width=wg_width, space_slice=space_slice
        )
    source = insert_mode(
        omega, dl, input_slice.x, input_slice.y, epsr, m=num_input_modes
    )
    simulation = fdfd_ez(omega, dl, epsr, [Npml, Npml])
    Hx, Hy, Ez = simulation.solve(source)

    input = source[input_slice.x, input_slice.y]
    output = Ez[output_slice.x, output_slice.y]

    return epsr, input, output


if __name__ == "__main__":
    # generating data first
    epsr_arr = []
    input_arr = []
    output_arr = []
    for i in range(1000):
        epsr, input, output = generate_new_device()
        epsr_arr.append(epsr)
        input_arr.append(input)
        output_arr.append(output)
        if i % 10 == 0:
            print(f"Generated {i} devices")

    epsrs = t.stack(epsr_arr)
    inputs = t.stack(input_arr)
    outputs = t.stack(output_arr)

    # split the data into training and testing
    dataset = TensorDataset(epsrs, inputs, outputs)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # initialize the model
    model = OutputPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # hyperparams
    num_epochs = 10

    # training loop
    for epoch in range(num_epochs):
        running_loss = 0
        for i, data in enumerate(train_loader, 0):
            epsr, input, output = data
            optimizer.zero_grad()

            # forward
            outputs = model(epsr, input)
            loss = criterion(outputs, output)

            # backprop
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10}")
                running_loss = 0

    # test on the test set
    with t.no_grad():
        for i, data in enumerate(test_loader, 0):
            epsr, input, output = data
            outputs = model(epsr, input)
            loss = criterion(outputs, output)
            print(f"Test loss: {loss}")

    t.save(model.state_dict(), "fdtd.pth")
