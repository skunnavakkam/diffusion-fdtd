# in a 100 x 100 array
# we then generate slices
# then based on those slices, where we see the inputs

import numpy as np
import matplotlib.pyplot as plt
from perlin_numpy import generate_perlin_noise_2d


import numpy as np
import autograd.numpy as npa

import copy

import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 100

import matplotlib.pylab as plt

from autograd.scipy.signal import convolve as conv

import ceviche
from ceviche import fdfd_ez, jacobian
from ceviche.optimizers import adam_optimize
from ceviche.modes import insert_mode

import collections

# Create a container for our slice coords to be used for sources and probes
# this is going to be super basic, where we take one input and one output to train
Slice = collections.namedtuple("Slice", "x y")

Nx = Ny = 100
Npml = 10
space = 10
wg_width = 12
epsr_init = 12
dev_Nx = dev_Ny = 60
space_slice = 8
omega = 2 * np.pi * 200e12
dl = 40e-9


def generate_modes():
    pass


def generate_structure(Nx, Ny, dev_Nx, dev_Ny):
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

    return epsr, bg_epsr, design_region, input_slice, output_slice


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


def viz_sim_tile(epsr, source, ax, slices=[]):
    """Solve and visualize a simulation with permittivity 'epsr' on given axes."""
    simulation = fdfd_ez(omega, dl, epsr, [Npml, Npml])
    Hx, Hy, Ez = simulation.solve(source)
    ceviche.viz.real(Ez, outline=epsr, ax=ax[0], cbar=False)
    for sl in slices:
        ax[0].plot(sl.x * np.ones(len(sl.y)), sl.y, "b-")
    ceviche.viz.abs(epsr, ax=ax[1], cmap="Greys")


def generate_structure_test():
    treshold = 0.05
    device = generate_perlin_noise_2d((dev_Nx, dev_Ny), (2, 2))
    return device > treshold


def output_vector(structure, modes):
    pass


def mask_combine_epsr(epsr, bg_epsr, design_region):
    """Utility function for combining the design region epsr and the background epsr"""
    return epsr * design_region + bg_epsr * (design_region == 0).astype(float)


if __name__ == "__main__":
    fig, axs = plt.subplots(3, 3, figsize=(100, 100))

    for i in range(3):
        for j in range(3):
            ax_pair = [axs[i, j], axs[i, j]]
            epsr, bg_epsr, design_region, input_slice, output_slice = (
                generate_structure(Nx, Ny, dev_Nx, dev_Ny)
            )

            epsr_total = mask_combine_epsr(epsr, bg_epsr, design_region)

            # set the middle 40 x 40 to be the noise structure
            epsr_total[20:80, 20:80] = generate_structure_test() * epsr_init

            # Setup source
            source = insert_mode(
                omega, dl, input_slice.x, input_slice.y, epsr_total, m=1
            )

            # Setup probe
            probe = insert_mode(
                omega, dl, output_slice.x, output_slice.y, epsr_total, m=2
            )

            # Simulate initial device
            viz_sim_tile(
                epsr_total, source, ax_pair, slices=[input_slice, output_slice]
            )

    plt.show()
