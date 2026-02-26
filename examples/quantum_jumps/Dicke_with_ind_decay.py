import matplotlib.pyplot as plt
import os

from qutip import (
    Options,
    basis,
    mcsolve,
    qeye,
    sigmam,
    sigmax,
    sigmaz,
    tensor,
)
import numpy as np



# Construct local operator in 2^N dimensional Hilbert space
def local_op(single_site_op, n_sites, site):
    factors = [qeye(2) for _ in range(n_sites)]
    factors[site] = single_site_op
    return tensor(factors)

# Construct sum of local operators over all sites in 2^N dimensional Hilbert space
def local_op_sum(single_site_op, n_sites):
    total = None
    for site in range(n_sites):
        factors = [qeye(2) for _ in range(n_sites)]
        factors[site] = single_site_op
        op = tensor(factors)
        total = op if total is None else total + op
    return total





def main():
    N = 18
    g1D = 10
    g0 = 1
    N_TRAJ = 30
    wc = g1D * (N - 1) / 4
    wx = 0.73 * wc

    sz = local_op_sum(sigmaz(), N)
    sx = local_op_sum(sigmax(), N)
    sm = local_op_sum(sigmam(), N)

    ham_mc = wx * sx
    c_op_list = [np.sqrt(g0) * local_op(sigmam(), N, site) for site in range(N)]
    c_op_list.append(np.sqrt(g1D) * sm)

    psi0 = tensor([basis(2, 1) for _ in range(N)])

    t1 = np.linspace(0, 70, 140)
    num_cpus = max(1, min(N_TRAJ, os.cpu_count() or 1))
    mc_result = mcsolve(
        ham_mc,
        psi0,
        t1,
        c_op_list,
        [sz, sm.dag() * sm],
        ntraj=N_TRAJ,
        options=Options(num_cpus=num_cpus),
    )

    sz_per_spin = mc_result.expect[0] / N
    np.save("sz_val_example.npy", sz_per_spin)

    plt.figure(figsize=(10.5, 3))
    plt.plot(t1, sz_per_spin, "k-o")
    plt.xlabel("Time t")
    plt.ylabel(r"$\langle S_z \rangle / N$")
    plt.tight_layout()
    plt.savefig(f"sz_val_Ntr_{N_TRAJ}_N_{N}.pdf")


if __name__ == "__main__":
    main()



