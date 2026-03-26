import numpy as np
import matplotlib.pyplot as plt

def create_gnm_matrix_vectorized(positions):
    """Creates the Green's function matrix using vectorized operations."""
    N = len(positions)
    
    # Compute all pairwise displacement vectors using broadcasting
    # Shape: (N, N, 3)
    r_vec = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    
    # Compute Euclidean distances
    # Shape: (N, N)
    #R = np.linalg.norm(r_vec, axis=2)
    R = calculate_distances_chunk(positions)
    # Set diagonal elements to a non-zero value to avoid division by zero
    # We'll zero them out later
    np.fill_diagonal(R, 1.0)
    
    # Compute kr
    kr = 2 * np.pi * R
    
    # Create e_0 vector (polarization)
    e_0 = (1 / np.sqrt(2)) * np.array([1, 1j, 0])
    #e_0 = np.array([0, 0, 1])
    # Compute dot products between e_0 and all displacement vectors
    # Shape: (N, N)
    dot_products = np.sum(e_0 * r_vec, axis=2)
    epsilon = 1e-10  # Small value to prevent division by zero

    # Ensure R is non-zero
    R = np.where(R == 0, epsilon, R)
    # Compute cos^2(theta)
    # Shape: (N, N)
    cos2_theta = np.abs(dot_products / R) ** 2
    # Ensure kr is non-zero
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


#create an 3D array with x,y,z lattice with lattice spacing a_x, a_y, a_z
def create_lattice(nx, ny, nz, a_x, a_y, a_z):
    """Creates a 3D lattice of points with given dimensions and spacings."""
    x = np.arange(-nx/2, nx/2) * a_x
    y = np.arange(-ny/2, ny/2) * a_y
    z = np.arange(-nz/2, nz/2) * a_z
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    lattice = np.stack((xv, yv, zv), axis=-1).reshape(-1, 3)
    return lattice



def calculate_distances_chunk(positions, chunk_size=1000):
    """Calculate distances using a chunked approach to prevent memory issues."""
    n_points = positions.shape[0]
    distances = np.zeros((n_points, n_points), dtype=np.float64)
    
    print(f"Calculating distances for {n_points} points using chunked approach...")
    
    # Process the matrix in chunks to avoid memory issues
    for i in range(0, n_points, chunk_size):
        end_i = min(i + chunk_size, n_points)
        chunk_positions_i = positions[i:end_i]
        
        for j in range(0, n_points, chunk_size):
            end_j = min(j + chunk_size, n_points)
            chunk_positions_j = positions[j:end_j]
            
            # Calculate distances for this chunk
            r_vec = chunk_positions_i[:, np.newaxis, :] - chunk_positions_j[np.newaxis, :, :]
            
            # Use a safer norm calculation to avoid MKL errors
            # Square each component first, then sum, then sqrt
            r_squared = np.sum(r_vec**2, axis=2)
            chunk_distances = np.sqrt(np.maximum(r_squared, 0.0))  # Ensure non-negative before sqrt
            
            # Store in the full distance matrix
            distances[i:end_i, j:end_j] = chunk_distances
            
            #print(f"Processed chunk ({i}:{end_i}, {j}:{end_j})")
    
    return distances

if __name__ == "__main__":
    nx, ny, nz = 5, 5, 1000
    a_x, a_y, a_z = 0.5, 0.5, 1.0

    positions = create_lattice(nx, ny, nz, a_x, a_y, a_z)

    gnm, dnm = create_gnm_matrix_vectorized(positions)

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    im1 = axs.imshow(np.abs(gnm), cmap='hot', interpolation='nearest')
    cbar1 = plt.colorbar(im1, ax=axs, fraction=0.046, pad=0.04)
    cbar1.set_label(r'$|G_{nm}|$')
    axs.set_xlabel('Atom $n$')
    axs.set_ylabel('Atom $m$')

    
    plt.tight_layout()
    plt.savefig(f'gnm_matrix_{nx}_{ny}_{nz}.pdf')