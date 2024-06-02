import ceviche
from ceviche import fdfd_ez
from ceviche.modes import insert_mode
from collections import namedtuple
import numpy as np
from perlin_numpy import generate_perlin_noise_2d
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as npa
from random import randint

Slice = namedtuple("Slice", "x y")
Nx = Ny = 100
Npml = 10
space = 10
wg_width = 12
epsr_init = 12
dev_Nx = dev_Ny = 60
space_slice = 4
omega = 2 * np.pi * 200e12
dl = 40e-9


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
