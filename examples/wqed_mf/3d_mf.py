import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path


#create an 3D array with x,y,z lattice with lattice spacing a_x, a_y, a_z
def create_lattice(nx, ny, nz, a_x, a_y, a_z):
    """Creates a 3D lattice of points with given dimensions and spacings."""
    x = np.arange(-nx/2, nx/2) * a_x
    y = np.arange(-ny/2, ny/2) * a_y
    z = np.arange(-nz/2, nz/2) * a_z
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    lattice = np.stack((xv, yv, zv), axis=-1).reshape(-1, 3)
    return lattice

def create_gnm_matrix_vectorized(positions):
    """Creates the Green's function matrix using vectorized operations."""
    N = len(positions)
    
    # Compute all pairwise displacement vectors using broadcasting
    # Shape: (N, N, 3)
    r_vec = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    
    # Compute Euclidean distances
    # Shape: (N, N)
    R = np.linalg.norm(r_vec, axis=2)
    
    # Set diagonal elements to a non-zero value to avoid division by zero
    np.fill_diagonal(R, 1.0)
    kr = 2 * np.pi * R
    
    # Create e_0 vector (polarization)
    e_0 = (1 / np.sqrt(2)) * np.array([1, 1j, 0])
    #e_0 = np.array([0, 0, 1])
    # Compute dot products between e_0 and all displacement vectors
    # Shape: (N, N)
    dot_products = np.sum(e_0 * r_vec, axis=2)
    
    # Compute cos^2(theta)
    # Shape: (N, N)
    cos2_theta = np.abs(dot_products / R) ** 2
    
    # Compute the Green's function
    g1 = np.exp(1j * kr) / kr * (
        (1 + (1j * kr - 1) / kr**2) + cos2_theta * (-1 + (3 - 3 * 1j * kr) / kr**2))
    
    # Set diagonal elements to zero (no self-interaction)
    np.fill_diagonal(g1, 0)
    
    # Compute gnm and dnm
    dnm = -3/4 * np.real(g1)
    gnm = 3/2 * np.imag(g1)
    #dnm = np.zeros((N,N))
    
    return gnm, dnm