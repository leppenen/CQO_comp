#!/usr/bin/env python3
"""
WQED Mean-Field Example for HPC Cluster (parallelized)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm


def create_1d_gnm_matrix_vectorized(positions, gamma1D=1):
    """Creates the Green's function matrix for 1D lattice using vectorized operations."""
    N = len(positions)
    
    # Compute all pairwise displacement vectors in 1D
    r_vec = positions[:, np.newaxis] - positions[np.newaxis, :]
    # Compute Euclidean distances
    R = np.abs(r_vec)
    # Set diagonal elements to avoid division by zero
    np.fill_diagonal(R, 1.0)
    # Compute kr
    kr = 2 * np.pi * R
    # Compute the 1D Green's function
    g1 = gamma1D / 2 * np.exp(1j * kr)
    np.fill_diagonal(g1, 0)
    
    gnm = 2 * np.real(g1)
    dnm = np.imag(g1)
    
    return gnm, dnm


def system_odes_prop(t, y, N, gamma, Dnm, Gnm, Omega):
    """Mean-field ODE system for collective atom-photon interactions."""
    sx_vals = y[:N]
    sy_vals = y[N:2*N]
    sz_vals = y[2*N:3*N]

    Dnmx = np.dot(Dnm, sx_vals)
    Dnmy = np.dot(Dnm, sy_vals)
    Gnmx = np.dot(Gnm, sx_vals)
    Gnmy = np.dot(Gnm, sy_vals)

    dsx_dt = -gamma/2 * sx_vals - 2 * np.imag(Omega) * sz_vals + sz_vals/2 * Gnmx + sz_vals * Dnmy
    dsy_dt = -gamma/2 * sy_vals - 2 * np.real(Omega) * sz_vals + sz_vals/2 * Gnmy - sz_vals * Dnmx
    dsz_dt = -gamma * (1 + sz_vals) + 2 * sy_vals * np.real(Omega) + 2 * sx_vals * np.imag(Omega) - 1/2 * sx_vals * Gnmx - 1/2 * sy_vals * Gnmy + sy_vals * Dnmx - sx_vals * Dnmy
    
    return np.concatenate([dsx_dt, dsy_dt, dsz_dt])


def solve_system_prop(N, gamma, Dnm, Gnm, Omega_n, t_span, y0, n_timepoints=50):
    """Solve ODE system for a single Omega value."""
    t_eval = np.linspace(t_span[0], t_span[1], n_timepoints)
    sol = solve_ivp(
        system_odes_prop,
        t_span,
        y0,
        args=(N, gamma, Dnm, Gnm, Omega_n),
        t_eval=t_eval,
        rtol=1e-7,
        atol=1e-9
    )
    return sol


def solve_system_wrapper(args):
    """Wrapper for parallelization."""
    return solve_system_prop(*args)


def parallel_solve_omega(N, gamma, Dnm, Gnm, Omega_n_values, t_span, y0, n_workers=4):
    """Parallelized execution for solving ODEs across different Omega values."""
    with mp.Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(
                solve_system_wrapper,
                [(N, gamma, Dnm, Gnm, Omega_n, t_span, y0) for Omega_n in Omega_n_values],
                chunksize=2
            ),
            total=len(Omega_n_values),
            desc="Solving ODE sweep"
        ))
    return results



def main():
    # Parameters
    nz = 100
    a_z = 0.9
    eta = 0.0
    g1d = 10.0
    gamma_decay = 1.0
    z0 = -1.0
    gammap = g1d + gamma_decay
    t_final = 10.0 / (nz * g1d)
    n_workers = 4
    
    output_dir = Path("results/wqed_cluster")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create lattice
    z_regular = np.arange(-nz/2, nz/2) * a_z
    z_disorder = np.random.normal(0, eta / (2 * np.pi), nz)
    z = z_regular + z_disorder

    # Green's function matrices
    omc = g1d * (nz - 1) / 4
    Gnm, Dnm = create_1d_gnm_matrix_vectorized(z, g1d)

    # Parameter sweep
    r_vals = np.linspace(0.0, 1.2, 20)
    Omega_n_values = [r * omc * np.exp(2j * np.pi * z) for r in r_vals]

    # Initial condition
    t_span = (0, t_final)
    y0 = np.zeros(3 * nz)
    y0[2 * nz:] = z0

    # Solve in parallel
    sol_vectors = parallel_solve_omega(nz, gammap, Dnm, Gnm, Omega_n_values, t_span, y0, n_workers)

    # Extract observables
    sz_values = np.array([sol.y[2 * nz:, -1].mean() for sol in sol_vectors])
    s2_values = np.array([
        np.abs(((sol.y[0:nz, -1] - 1j * sol.y[nz:2*nz, -1]) * np.exp(-2j * np.pi * z)).mean())
        for sol in sol_vectors
    ])

    # Save results
    np.save(output_dir / f"results_nz{nz}_az{a_z:.1f}_eta{eta:.2f}.npy", 
            {'r_vals': r_vals, 'sz_values': sz_values, 's2_values': s2_values}, 
            allow_pickle=True)

    # Save CSV
    with open(output_dir / f"summary_nz{nz}_az{a_z:.1f}_eta{eta:.2f}.csv", "w") as f:
        f.write("r,sz_mean,s2_abs\n")
        for r, sz, s2 in zip(r_vals, sz_values, s2_values):
            f.write(f"{r:.6f},{sz:.10e},{s2:.10e}\n")

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    axs[0].plot(r_vals, sz_values, 'b-o', linewidth=2, markersize=6)
    axs[0].set_ylabel(r'$\langle s_z \rangle$', fontsize=14)
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(r_vals, s2_values / 2, 'r-o', linewidth=2, markersize=6)
    axs[1].set_ylabel(r'$|\langle s_- e^{-i k z_n} \rangle|$', fontsize=14)
    axs[1].set_xlabel(r'$\Omega/\Omega_c$', fontsize=14)
    axs[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(output_dir / f"plot_nz{nz}.pdf", bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Done! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
