import numpy as np
import multiprocessing as mp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
#Analytical cases
from bistability import scan_sz_vs_r,scan_sz_sy_vs_r
from propogation1D import sz_vals_1D
plt.style.use(['my_science.mplstyle'])

def create_1d_gnm_matrix_vectorized(positions,gamma1D = 1):
    """Creates the Green's function matrix for 1D lattice using vectorized operations."""
    N = len(positions)
    
    # Compute all pairwise displacement vectors in 1D
    # Shape: (N, N)
    r_vec = positions[:, np.newaxis] - positions[np.newaxis, :]
    # Compute Euclidean distances
    # Shape: (N, N)
    R = np.abs(r_vec)
    # Set diagonal elements to a non-zero value to avoid division by zero
    # We'll zero them out later
    np.fill_diagonal(R, 1.0)
    # Compute kr
    kr = 2 * np.pi * R
    # Compute the 1D Green's function
    g1 = gamma1D/2*np.exp(1j*kr)
    np.fill_diagonal(g1, 0)
    
    gnm = 2*np.real(g1)
    dnm = np.imag(g1)
    
    return gnm, dnm

def system_odes_prop(t,y,N,gamma,Dnm,Gnm,Omega):
    """Creates the system of the mean-field equations for any dimension"""
    sx_vals = y[:N]
    sy_vals = y[N:2*N]
    sz_vals = y[2*N:3*N]

    Dnmx = np.dot(Dnm, sx_vals)
    Dnmy = np.dot(Dnm, sy_vals)
    Gnmx = np.dot(Gnm, sx_vals)
    Gnmy = np.dot(Gnm, sy_vals)

    dsx_dt = -gamma/2 * sx_vals - 2 * np.imag(Omega) * sz_vals + sz_vals/2 * Gnmx + sz_vals * Dnmy
    dsy_dt = -gamma/2 * sy_vals - 2 * np.real(Omega) * sz_vals + sz_vals/2 * Gnmy - sz_vals * Dnmx
    dsz_dt = -gamma * (1 + sz_vals) + 2 * sy_vals * np.real(Omega) + 2 * sx_vals * np.imag(Omega)   - 1/2 * sx_vals * Gnmx - 1/2 * sy_vals * Gnmy + sy_vals*Dnmx - sx_vals* Dnmy
    return np.concatenate([dsx_dt, dsy_dt, dsz_dt])

def solve_system_prop(N, gamma, Dnm, Gnm, Omega_n, t_span, y0):
    """ODE solution"""
    sol = solve_ivp(system_odes_prop, t_span, y0, args=(N, gamma, Dnm, Gnm, Omega_n),  t_eval= np.linspace(t_span[0],t_span[1],50))
    return sol


###This is parallelization
def solve_system_wrapper(args):
    return solve_system_prop(*args)

def parallel_solve_omega(N, gamma, Dnm, Gnm,  Omega_n_values, t_span, y0):
    """Parallelized execution for solving ODEs across different Omega values."""
    with mp.Pool(processes=5) as pool:
        results = list(tqdm(pool.imap(solve_system_wrapper, 
                                      [(N, gamma, Dnm, Gnm,  Omega_n, t_span, y0) for Omega_n in Omega_n_values], chunksize=4), 
                            total=len(Omega_n_values)))
    return results

def main():
    # Parameters for kx = ky = 1 case only
    0
    
    folder_name = "dicke_time/"
    # Parameters that remain constant
    nz = 200
    a_z = 0.9
    eta = 0  # disorder strength
    gamma = 1
    z0 = -1
    g1D = 10
    gammap = g1D+gamma
    t_final = 10/(nz*g1D)
    
    print(f"\n{'='*60}")
    print(f"Running for t_final = {t_final}")
    print(f"{'='*60}")
    z_regular = np.arange(-nz/2, nz/2) * a_z

    # Create 1D lattice with disorder
    # z_i = a_z * i + delta_z_i
    # where <delta_z_i * delta_z_j> = (eta/(2*pi))^2 * delta_ij
    z_disorder = np.random.normal(0, eta/(2*np.pi), nz)
    z = z_regular + z_disorder

    

    omc = g1D*(nz-1) / 4
    Gnm, Dnm = create_1d_gnm_matrix_vectorized(z, g1D)
    r_vals = np.linspace(0, 1.2, 20)
    Omega_n_values = []
    for r in r_vals:
        Omega = r * omc
        Omega_n_values.append(Omega * np.exp(2j * np.pi * z))

    print(f"Number of Omega values: {len(Omega_n_values)}")

    print(f"Total system size N: {nz}")

    t_span = (0, t_final)
    y0 = np.zeros(3 * nz)
    y0[2 * nz:] = z0

    # Solve in parallel
    sol_vectors = parallel_solve_omega(nz, gammap, Dnm, Gnm, Omega_n_values, t_span, y0)
    print(f"Number of solutions: {len(sol_vectors)}")

    # Generate filenames dynamically
    solution_filename = f"{folder_name}n{nz}_1d_{a_z}_eta_{eta}.npy"

    # Save results
    np.save(solution_filename, sol_vectors)
    print(f"Saved solutions to {solution_filename}")
    
    # Extract and plot z-values
    index_range = np.arange(nz)

    # Z steady state values
    z_values = np.array([sol.y[2*nz+index_range, -1] for sol in sol_vectors])

    # Extract spin component values
    sz_values = np.array([sol.y[2 * nz:, -1].mean() for sol in sol_vectors])
  # For s_- = sx - i*sy: compute as (1/N) * sum_n (sx_n - i*sy_n) * exp(2j*pi*z_n)
    s2_values = np.array([((sol.y[0: nz, -1] - 1j*sol.y[nz: 2*nz, -1]) * np.exp(-2j * np.pi * z)).mean() 
                          for sol in sol_vectors])
    s2_values = np.abs(s2_values)

    # Plot the colormap
    plt.figure(figsize=(8, 6))
    plt.imshow(np.flipud(z_values), aspect='auto', cmap='viridis',
            extent=[0, nz, r_vals[0], r_vals[-1]])

    plt.colorbar(label="$s_n^z$")
    plt.xlabel("$n_z$")
    plt.ylabel("$\\Omega/\\Omega_\\text{c}$")
    plt.savefig(f'{folder_name}/n{nz}_1d_{a_z}_eta_{eta}.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {folder_name}/n{nz}_1d_{a_z}_eta_{eta}.pdf")
    


    
    # Create summary plots
    fig, axs = plt.subplots(2, 1, figsize=(5.5, 8))
    
    axs[0].plot(r_vals, sz_values, 'b-o')
    axs[0].set_ylabel('$\\langle s_z \\rangle$')

    #Compare with cavity solution 
    r1_vals, sz1_vals, sy1_vals = scan_sz_sy_vs_r(g1D*(nz-1), np.sin(2*np.pi*a_z), gammap, r_min=r_vals[0], r_max=r_vals[-1], npts=1000)
    axs[0].scatter(r1_vals,sz1_vals, s = 1,label = 'Cavity')
    #Compare with Dicke solution
    axs[0].plot(r1_vals,-np.sqrt(1-r1_vals**2), 'm-',label = 'Dicke')
    #Compare with forward propogating solution 
    sz1_prop = np.zeros_like(r1_vals)
    sm1_prop = np.zeros_like(r1_vals)
    for i, r in enumerate(r1_vals):
        Omega1 = r*omc
        sz1_prop[i] = np.mean(sz_vals_1D(Omega1,g1D,gammap,z)[0])
        sm1_prop[i] = np.mean(sz_vals_1D(Omega1,g1D,gammap,z)[1])
    axs[0].plot(r1_vals,sz1_prop,'k-',label = 'Forward Prop')
    axs[0].legend(fontsize = 12)
        


    
    axs[1].plot(r_vals, s2_values/2, 'b-o')
    axs[1].set_ylabel('$|\\langle s_- e^{-i k z_n} \\rangle|$')
    axs[1].set_xlabel('$\\Omega/\\Omega_\\text{c}$')
    axs[1].plot(r1_vals,sm1_prop,'k-',label = 'Forward Prop')
    axs[1].scatter(r1_vals,sy1_vals,s = 1,label = 'Cavity')
    axs[1].legend(fontsize = 12)
    fig.suptitle(f'$N = {nz}, a_z = {a_z}, \eta = {eta}$', fontsize = 20)
    plt.tight_layout()
    plt.savefig(f"{folder_name}/steady_states_1d_{nz}_az_{a_z}_eta_{eta}.pdf")
    plt.close()
    
    # Save summary data
    np.save(f"{folder_name}/sz_values_1d_{nz}_az_{a_z}_eta_{eta}.npy", sz_values)
    np.save(f"{folder_name}/s2_values_1d_{nz}_az_{a_z}_eta_{eta}.npy", s2_values)
    print(f"Results saved successfully")


if __name__ == "__main__":
    main()
