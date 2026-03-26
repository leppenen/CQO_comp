import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.sparse.linalg import LinearOperator, eigsh
except Exception:
    LinearOperator = None
    eigsh = None


def create_lattice(nx, ny, nz, a_x, a_y, a_z):
    """Create a 3D rectangular lattice of atomic positions."""
    x = np.arange(-nx / 2, nx / 2) * a_x
    y = np.arange(-ny / 2, ny / 2) * a_y
    z = np.arange(-nz / 2, nz / 2) * a_z
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    return np.stack((xv, yv, zv), axis=-1).reshape(-1, 3)


def parse_int_list(text):
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def parse_nxy_list(text):
    """
    Parse Nx,Ny pairs.
    Accepted tokens: "4" (means 4x4), "4x6", "4*6", "4:6".
    Example: "3,4,5x7"
    """
    pairs = []
    for token in text.split(","):
        t = token.strip().lower()
        if not t:
            continue
        if "x" in t:
            nx, ny = t.split("x", maxsplit=1)
            pairs.append((int(nx), int(ny)))
        elif "*" in t:
            nx, ny = t.split("*", maxsplit=1)
            pairs.append((int(nx), int(ny)))
        elif ":" in t:
            nx, ny = t.split(":", maxsplit=1)
            pairs.append((int(nx), int(ny)))
        else:
            n = int(t)
            pairs.append((n, n))
    return pairs


def _compute_gnm_block(pos_i, pos_j, e0, i0, j0):
    """Compute a dense block of Gnm between two position chunks."""
    r_vec = pos_i[:, np.newaxis, :] - pos_j[np.newaxis, :, :]
    r2 = np.sum(r_vec * r_vec, axis=2)
    r = np.sqrt(np.maximum(r2, 0.0))

    # Protect divisions; exact self-interactions are zeroed afterwards.
    r_safe = np.where(r == 0.0, 1.0, r)
    kr = 2.0 * np.pi * r_safe

    dot_products = np.sum(e0 * r_vec, axis=2)
    cos2_theta = np.abs(dot_products / r_safe) ** 2

    g1 = np.exp(1j * kr) / kr * (
        (1.0 + (1j * kr - 1.0) / (kr**2))
        + cos2_theta * (-1.0 + (3.0 - 3.0 * 1j * kr) / (kr**2))
    )

    # Zero out all exact coincident points.
    g1[r == 0.0] = 0.0

    gnm_block = 1.5 * np.imag(g1)

    # Set spontaneous-emission term on diagonal.
    if i0 == j0:
        np.fill_diagonal(gnm_block, 1.0)

    return gnm_block


def build_gnm_linear_operator(positions, block_size=128):
    """Build a matrix-free linear operator y = Gnm @ x."""
    n = positions.shape[0]
    e0 = (1.0 / np.sqrt(2.0)) * np.array([1.0, 1j, 0.0], dtype=np.complex128)

    def matvec(x):
        x = np.asarray(x, dtype=np.float64)
        y = np.zeros(n, dtype=np.float64)

        for i0 in range(0, n, block_size):
            i1 = min(i0 + block_size, n)
            pos_i = positions[i0:i1]
            yi = np.zeros(i1 - i0, dtype=np.float64)

            for j0 in range(0, n, block_size):
                j1 = min(j0 + block_size, n)
                pos_j = positions[j0:j1]

                g_block = _compute_gnm_block(pos_i, pos_j, e0, i0, j0)
                yi += g_block @ x[j0:j1]

            y[i0:i1] = yi

        return y

    if LinearOperator is None:
        raise ImportError("scipy is required for matrix-free Lanczos mode (eigsh).")

    return LinearOperator(shape=(n, n), matvec=matvec, dtype=np.float64)


def largest_eig_dense(positions, block_size=128):
    """Exact largest eigenvalue via dense matrix assembly + eigvalsh."""
    n = positions.shape[0]
    e0 = (1.0 / np.sqrt(2.0)) * np.array([1.0, 1j, 0.0], dtype=np.complex128)
    gnm = np.zeros((n, n), dtype=np.float64)

    for i0 in range(0, n, block_size):
        i1 = min(i0 + block_size, n)
        pos_i = positions[i0:i1]
        for j0 in range(0, n, block_size):
            j1 = min(j0 + block_size, n)
            pos_j = positions[j0:j1]
            gnm[i0:i1, j0:j1] = _compute_gnm_block(pos_i, pos_j, e0, i0, j0)

    gnm = 0.5 * (gnm + gnm.T)
    return float(np.linalg.eigvalsh(gnm)[-1])


def largest_eig_lanczos(positions, block_size=128, tol=1e-5, maxiter=200):
    """Largest algebraic eigenvalue using matrix-free Lanczos."""
    op = build_gnm_linear_operator(positions, block_size=block_size)
    vals, _ = eigsh(op, k=1, which="LA", tol=tol, maxiter=maxiter)
    return float(vals[0])


def run_scan(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nxy_pairs = parse_nxy_list(args.nxy)
    nz_values = parse_int_list(args.nz)

    rows = []
    for nz in nz_values:
        print(f"\\n=== Nz={nz} ===")
        for nx, ny in nxy_pairs:
            n_atoms = nx * ny * nz
            if n_atoms > args.max_atoms:
                print(f"Skipping nx={nx}, ny={ny}, nz={nz} (N={n_atoms} > max_atoms={args.max_atoms})")
                continue

            positions = create_lattice(nx, ny, nz, args.ax, args.ay, args.az)
            t0 = time.time()

            if args.method == "dense":
                lam_max = largest_eig_dense(positions, block_size=args.block_size)
            else:
                lam_max = largest_eig_lanczos(
                    positions,
                    block_size=args.block_size,
                    tol=args.tol,
                    maxiter=args.maxiter,
                )

            dt = time.time() - t0
            area = nx * ny
            print(
                f"nx={nx:3d}, ny={ny:3d}, nz={nz:4d}, Nxy={area:4d}, "
                f"N={n_atoms:7d}, lambda_max={lam_max: .6e}, time={dt: .1f}s"
            )

            rows.append(
                {
                    "nx": nx,
                    "ny": ny,
                    "nz": nz,
                    "nxy": area,
                    "n_atoms": n_atoms,
                    "ax": args.ax,
                    "ay": args.ay,
                    "az": args.az,
                    "method": args.method,
                    "lambda_max": lam_max,
                    "time_sec": dt,
                }
            )

    if not rows:
        print("No data points were computed. Check --max-atoms and scan ranges.")
        return

    npz_path = out_dir / args.npz_name
    np.savez(
        npz_path,
        nx=np.array([r["nx"] for r in rows], dtype=np.int32),
        ny=np.array([r["ny"] for r in rows], dtype=np.int32),
        nz=np.array([r["nz"] for r in rows], dtype=np.int32),
        nxy=np.array([r["nxy"] for r in rows], dtype=np.int32),
        n_atoms=np.array([r["n_atoms"] for r in rows], dtype=np.int32),
        lambda_max=np.array([r["lambda_max"] for r in rows], dtype=np.float64),
        time_sec=np.array([r["time_sec"] for r in rows], dtype=np.float64),
        ax=np.array([args.ax], dtype=np.float64),
        ay=np.array([args.ay], dtype=np.float64),
        az=np.array([args.az], dtype=np.float64),
        method=np.array([args.method]),
    )
    print(f"Saved NPZ data: {npz_path}")

    fig, ax = plt.subplots(figsize=(8, 4))
    for nz in nz_values:
        series = [r for r in rows if r["nz"] == nz]
        if not series:
            continue
        series = sorted(series, key=lambda r: r["nxy"])
        x = [r["nxy"] for r in series]
        y = [r["lambda_max"] for r in series]
        ax.plot(x, y, marker="o", linewidth=3, label=f"Nz={nz}")

    ax.set_xlabel(r"$N_x N_y$")
    ax.plot(x,3*np.asarray(x)/4, "k--",linewidth=3 ,label=r"$3M/4$")
    ax.set_ylabel(r"max eigenvalue of $\gamma_{nm}$")
    ax.legend()
    fig.tight_layout()

    fig_path = out_dir / args.fig_name
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {fig_path}")


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Scan largest eigenvalue of Gnm vs Nx*Ny for multiple Nz at fixed ax, ay, az. "
            "Default mode uses matrix-free Lanczos (scipy eigsh)."
        )
    )
    p.add_argument("--nxy", default="2,3,4,5,6", help="Nx,Ny list, e.g. '3,4,5x7'")
    p.add_argument("--nz", default="50,100,200", help="Nz list, e.g. '50,100,200'")
    p.add_argument("--ax", type=float, default=0.5, help="Lattice spacing ax/lambda")
    p.add_argument("--ay", type=float, default=0.5, help="Lattice spacing ay/lambda")
    p.add_argument("--az", type=float, default=1.0, help="Lattice spacing az/lambda")

    p.add_argument("--method", choices=["lanczos", "dense"], default="lanczos")
    p.add_argument("--block-size", type=int, default=128, help="Chunk size for block matvec")
    p.add_argument("--tol", type=float, default=1e-5, help="Lanczos tolerance")
    p.add_argument("--maxiter", type=int, default=200, help="Lanczos max iterations")

    p.add_argument("--max-atoms", type=int, default=15000, help="Skip points with N > max-atoms")
    p.add_argument("--output-dir", default="gf_eigs", help="Output directory")
    p.add_argument("--npz-name", default="max_eig_scan.npz", help="Output NPZ filename")
    p.add_argument("--fig-name", default="max_eig_vs_nxny.pdf", help="Output figure filename")
    return p


if __name__ == "__main__":
    parser = build_parser()
    run_scan(parser.parse_args())
