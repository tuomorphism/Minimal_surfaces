"""Render every figure used by minimal_surfaces.ipynb into assets/.

The notebook embeds these as images rather than recomputing them, so it reads
end-to-end without waiting on a solver. Re-run this after changing anything that
affects the results:

    python scripts/render_assets.py            # everything
    python scripts/render_assets.py gallery    # one figure by name

Takes a few minutes; most of it is the refinement study and the gallery.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import curves, extract, spectral  # noqa: E402
from src.grid import Grid  # noqa: E402
from src.initial_guess import compute_initial_guess  # noqa: E402
from src.plateau import solve_plateau  # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "assets"
RADIUS = 0.3
FIELD_CMAP = "magma"
SURFACE_CMAP = "viridis"
CURVE_COLOR = "#e8482c"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 9,
        "axes.titlesize": 10,
        "figure.facecolor": "white",
        "savefig.bbox": "tight",
    }
)

_cache = {}


def circle(t):
    return curves.circle(t, radius=RADIUS)


def solved(name, gamma, resolution=64, **kw):
    """Solve once and reuse; several figures share the same solution."""
    key = (name, resolution)
    if key not in _cache:
        print(f"  solving {name} at N={resolution} ...", flush=True)
        _cache[key] = solve_plateau(gamma, resolution=resolution, rtol=1e-3, **kw)
    return _cache[key]


def draw_curve(ax, gamma, num=400, **kw):
    for poly in curves.as_polylines(gamma, num):
        closed = np.vstack([poly, poly[:1]])
        ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=CURVE_COLOR, lw=2.2, **kw)


def equal_aspect(ax, points, margin=1.15, zoom=1.7):
    """Force a physically isometric view. Returns the (centre, half-range) used.

    Without this, a flat surface autoscales its degenerate axis to the data range
    and matplotlib stretches that to fill the box -- turning one cell of
    marching-cubes wobble at the rim into a dramatic (and completely spurious)
    curtain. Equal ranges on all three axes are the only honest way to draw a
    surface whose whole point is its shape.
    """
    pts = np.vstack(points)
    center = 0.5 * (pts.max(axis=0) + pts.min(axis=0))
    half = margin * 0.5 * (pts.max(axis=0) - pts.min(axis=0)).max()
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(c - half, c + half)
    ax.set_box_aspect([1, 1, 1], zoom=zoom)
    return center, half


def draw_surface(ax, solution, gamma, title, elev=26, azim=-58, clip=0.15):
    verts, faces = extract.extract_surface(solution, clip_fraction=clip)

    extent = curves.as_polylines(gamma, 200)
    if len(verts):
        extent = extent + [verts]
    ax.view_init(elev=elev, azim=azim)
    center, half = equal_aspect(ax, extent)

    if len(faces):
        # Colour by height against the *physical* z range rather than the data
        # range, so a flat surface comes out a single flat colour instead of a
        # bullseye made of sub-cell wobble.
        ax.plot_trisurf(
            verts[:, 0], verts[:, 1], faces, verts[:, 2],
            cmap=SURFACE_CMAP, linewidth=0, antialiased=True, alpha=0.95,
            vmin=center[2] - half, vmax=center[2] + half,
        )
    draw_curve(ax, gamma)
    ax.set_title(title)
    ax.set_axis_off()
    return verts, faces


def finish(fig, name):
    path = ASSETS / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ASSETS.parent)}")


# ---------------------------------------------------------------- figures


def fig_plateau_examples():
    """Intro: the disc for a circle, and a genuinely non-trivial case."""
    fig = plt.figure(figsize=(9.5, 3.7))
    for i, (name, gamma, title) in enumerate(
        [
            ("circle", circle, "$\\Gamma = \\mathbb{S}^1$: the flat disc"),
            ("trefoil", curves.trefoil, "trefoil knot: a Seifert-like surface"),
        ]
    ):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        draw_surface(ax, solved(name, gamma), gamma, title)
    finish(fig, "plateau_examples")


def fig_dirac_delta_forms():
    """The two objects the whole method rests on: delta_Gamma and delta_Sigma."""
    grid = Grid(48)
    guess = compute_initial_guess(grid, circle)
    solution = solved("circle", circle)

    fig = plt.figure(figsize=(10, 3.9))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    _quiver_of(ax, grid, guess.delta_gamma, threshold=0.06, stride=2, length=0.055)
    draw_curve(ax, circle)
    ax.set_title(r"$\delta_\Gamma$: a 2-form supported on the curve")
    ax.view_init(elev=20, azim=-60); ax.set_axis_off()
    equal_aspect(ax, curves.as_polylines(circle, 200), margin=1.8)

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    _quiver_of(ax, solution.grid, solution.eta, threshold=0.15, stride=3, length=0.075)
    draw_curve(ax, circle)
    ax.set_title(r"$\delta_\Sigma$: a 1-form supported on the surface")
    ax.view_init(elev=20, azim=-60); ax.set_axis_off()
    equal_aspect(ax, curves.as_polylines(circle, 200), margin=1.8)

    finish(fig, "dirac_delta_forms")


def _quiver_of(ax, grid, field, threshold, stride, length):
    mag = spectral.pointwise_norm(field)
    sel = mag > threshold * mag.max()
    sub = np.zeros_like(sel)
    sub[::stride, ::stride, ::stride] = True
    sel &= sub
    pts = grid.positions_grid[sel]
    vecs = field[sel]
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=-1, keepdims=True), 1e-30)
    colors = plt.get_cmap(FIELD_CMAP)(mag[sel] / mag.max())
    ax.quiver(
        pts[:, 0], pts[:, 1], pts[:, 2],
        vecs[:, 0], vecs[:, 1], vecs[:, 2],
        length=length, normalize=True, colors=colors, linewidth=0.7,
    )


def fig_initial_guess():
    """eta_0: the Biot-Savart field of the curve, before any optimization."""
    grid = Grid(64)
    guess = compute_initial_guess(grid, circle)
    mid = grid.resolution // 2
    xs = grid.positions_grid[:, mid, :, 0]
    zs = grid.positions_grid[:, mid, :, 2]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))

    m = spectral.pointwise_norm(guess.delta_gamma)[:, mid, :]
    axes[0].contourf(xs, zs, m, levels=60, cmap=FIELD_CMAP)
    axes[0].set_title(r"$|\delta_\Gamma|$  (slice $y=0.5$)")

    axes[1].contourf(xs, zs, guess.eta_0[:, mid, :, 2], levels=60, cmap="RdBu_r")
    axes[1].set_title(r"$z$-component of $\eta_0$")

    axes[2].contourf(xs, zs, spectral.pointwise_norm(guess.eta_0)[:, mid, :],
                     levels=60, cmap=FIELD_CMAP)
    axes[2].set_title(r"$|\eta_0|$ — spread out, not yet minimal")

    for ax in axes:
        ax.plot([0.5 - RADIUS, 0.5 + RADIUS], [0.5, 0.5], "o", color=CURVE_COLOR, ms=5)
        ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$z$")
    fig.tight_layout()
    finish(fig, "initial_guess")


def fig_operators():
    """Why forward differences: locality, and pairing with the Poisson solver."""
    N = 32
    grid = Grid(N)
    h = grid.h
    k = 2 * np.pi * np.fft.fftfreq(N, d=h)
    theta = k * h
    symbols = {
        "forward difference": (np.exp(1j * theta) - 1) / h,
        "midpoint (paper eq. 23)": 1j * np.sin(theta) / h,
        "spectral": 1j * k,
    }
    colors = {"forward difference": "#1b7f43", "midpoint (paper eq. 23)": "#c2761a",
              "spectral": "#3b5fc0"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    step = (np.arange(N) >= N // 2).astype(float)
    z = np.arange(N) * h
    for name, sym in symbols.items():
        d = np.fft.ifft(sym * np.fft.fft(step)).real
        width = int((np.abs(d) > 1e-8 * np.abs(d).max()).sum())
        axes[0].plot(z, np.abs(d) * h, color=colors[name], lw=1.6,
                     label=f"{name}\n  {width} cells, $L_1$={np.abs(d).sum()*h:.2f}")
    axes[0].set_title("Gradient of a step: how localized is $d$?")
    axes[0].set_xlabel("$z$"); axes[0].set_ylabel(r"$|d\phi| \cdot h$")
    axes[0].legend(fontsize=7.5, loc="upper left")

    order = np.argsort(k)
    seven_point = 4 / h**2 * np.sin(theta / 2) ** 2
    for name, sym in symbols.items():
        axes[1].plot(k[order], (np.abs(sym) ** 2)[order], color=colors[name], lw=1.6,
                     label=f"$|\\hat d|^2$, {name}")
    axes[1].plot(k[order], seven_point[order], "k--", lw=1.2,
                 label="paper's Algorithm 3 kernel")
    axes[1].set_title(r"$d^\top d$ vs the Poisson kernel it is inverted with")
    axes[1].set_xlabel("$k$"); axes[1].set_yscale("log"); axes[1].set_ylim(1, 1e5)
    axes[1].legend(fontsize=7.5, loc="lower center")

    fig.tight_layout()
    finish(fig, "operators")


def fig_admm(make_gif=True):
    """Convergence history, and the field collapsing onto the surface."""
    grid = Grid(48)
    exact = np.pi * RADIUS**2
    snapshots = {}
    frames = []
    want = {1, 2, 4, 12, 60}

    def callback(k, X, phi, history):
        mid = grid.resolution // 2
        sl = spectral.pointwise_norm(X)[:, mid, :]
        if k in want:
            snapshots[k] = sl.copy()
        if make_gif and (k <= 20 or k % 5 == 0) and k <= 200:
            frames.append((k, sl.copy(), history["mass"][-1]))

    print("  solving circle at N=48 with callback ...", flush=True)
    solution = solve_plateau(circle, resolution=48, rtol=1e-3, max_iter=600,
                             callback=callback)
    snapshots[solution.iterations] = spectral.pointwise_norm(solution.eta)[
        :, grid.resolution // 2, :
    ]

    # -- convergence curves
    h = solution.history
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    axes[0].plot(h["mass"], color="#1b7f43", lw=1.6)
    axes[0].axhline(exact, color="k", ls="--", lw=1.1, label=rf"exact $\pi r^2$ = {exact:.4f}")
    axes[0].set_xlabel("ADMM iteration"); axes[0].set_ylabel(r"$\|\eta\|_{\rm mass}$")
    axes[0].set_title("Mass converging to the disc area"); axes[0].legend(fontsize=8)
    axes[0].set_xscale("symlog")

    axes[1].semilogy(h["primal_relative"], label="primal feasibility", color="#1b7f43")
    axes[1].semilogy(h["dual_relative"], label="dual feasibility", color="#3b5fc0")
    axes[1].semilogy(h["criterion"], label="paper's $c$", color="#c2761a", ls=":")
    axes[1].set_xlabel("ADMM iteration"); axes[1].set_title("Residuals")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    finish(fig, "admm_convergence")

    # -- the field collapsing onto the disc
    keys = sorted(snapshots)
    fig, axes = plt.subplots(1, len(keys), figsize=(2.6 * len(keys), 3.0))
    vmax = max(s.max() for s in snapshots.values())
    xs = grid.positions_grid[:, 0, :, 0]
    zs = grid.positions_grid[:, 0, :, 2]
    for ax, k in zip(axes, keys):
        ax.contourf(xs, zs, snapshots[k], levels=50, cmap=FIELD_CMAP, vmin=0, vmax=vmax)
        ax.set_title(f"iteration {k}")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.plot([0.5 - RADIUS, 0.5 + RADIUS], [0.5, 0.5], "o", color=CURVE_COLOR, ms=4)
    fig.suptitle(r"$|\eta|$ on the slice $y=0.5$, collapsing onto the disc", y=1.02)
    fig.tight_layout()
    finish(fig, "admm_iterations")

    if make_gif and frames:
        _write_gif(grid, frames, exact, vmax)


def _write_gif(grid, frames, exact, vmax):
    xs = grid.positions_grid[:, 0, :, 0]
    zs = grid.positions_grid[:, 0, :, 2]
    fig, ax = plt.subplots(figsize=(4.0, 3.6))

    def draw(i):
        ax.clear()
        k, sl, mass = frames[i]
        ax.contourf(xs, zs, sl, levels=50, cmap=FIELD_CMAP, vmin=0, vmax=vmax)
        ax.plot([0.5 - RADIUS, 0.5 + RADIUS], [0.5, 0.5], "o", color=CURVE_COLOR, ms=5)
        ax.set_title(f"iteration {k}    mass = {mass:.4f}  (exact {exact:.4f})", fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=140)
    path = ASSETS / "admm_evolution.gif"
    anim.save(path, writer=animation.PillowWriter(fps=7))
    plt.close(fig)
    print(f"  wrote {path.relative_to(ASSETS.parent)}")


def fig_surface_extraction():
    """From the impulse eta to a mesh, via the level set with a unit jump."""
    solution = solved("circle", circle)
    grid = solution.grid
    u = extract.level_set(grid, solution.eta)
    mid = grid.resolution // 2

    fig = plt.figure(figsize=(12, 3.7))

    ax = fig.add_subplot(1, 3, 1)
    xs = grid.positions_grid[:, mid, :, 0]
    zs = grid.positions_grid[:, mid, :, 2]
    im = ax.contourf(xs, zs, u[:, mid, :], levels=60, cmap="RdBu_r")
    ax.set_title("$u$ (slice $y=0.5$)"); ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(1, 3, 2)
    profile = u[mid, mid, :]
    ax.plot(grid.positions_grid[mid, mid, :, 2], profile, color="#3b5fc0", lw=1.6)
    # The continuum theory gives a jump of exactly 1. Discretely it comes out a
    # little short: u only picks up the exact part of eta in the least-squares
    # sense, and the surface is smeared over about a cell. Report what is there.
    jump = profile.max() - profile.min()
    ax.set_title(f"jump across $\\Sigma$: {jump:.2f}  (continuum: 1)")
    ax.set_xlabel("$z$"); ax.grid(alpha=0.3)

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    draw_surface(ax, solution, circle, "extracted mesh")

    fig.tight_layout()
    finish(fig, "surface_extraction")


def fig_refinement():
    exact = np.pi * RADIUS**2
    Ns = [16, 24, 32, 48, 64]
    errs, errs_moll = [], []
    for N in Ns:
        errs.append(abs(solve_plateau(circle, resolution=N, rtol=1e-3, max_iter=600,
                                      sigma_cells=0.0).mass - exact))
        errs_moll.append(abs(solve_plateau(circle, resolution=N, rtol=1e-3,
                                           max_iter=600).mass - exact))
        print(f"  N={N}: {errs[-1]:.5f} / {errs_moll[-1]:.5f}", flush=True)
    h = 1.0 / np.array(Ns, dtype=float)

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.loglog(h, errs, "o-", color="#1b7f43", label="no mollification")
    ax.loglog(h, errs_moll, "s-", color="#3b5fc0", label="mollified (default)")
    ax.loglog(h, 0.5 * h * 2 * np.pi * RADIUS, "k--", lw=1.1,
              label=r"$\frac{h}{2}\times$ perimeter")
    ax.set_xlabel("$h$"); ax.set_ylabel(r"$|\;\|\eta\|_{\rm mass} - \pi r^2|$")
    ax.set_title("Error under refinement"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    finish(fig, "refinement")


def fig_gallery():
    entries = [
        ("circle", circle, "circle", dict()),
        ("ellipse", lambda t: curves.ellipse(t, a=0.35, b=0.2), "ellipse", dict()),
        ("trefoil", curves.trefoil, "trefoil knot", dict()),
        ("borromean", curves.borromean_rings(), "Borromean rings", dict(clip=0.22)),
    ]
    fig = plt.figure(figsize=(13, 3.1))
    for i, (name, gamma, title, kw) in enumerate(entries):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        sol = solved(name, gamma)
        draw_surface(ax, sol, gamma, f"{title}\n" + rf"$\|\eta\|_{{\rm mass}}$ = {sol.mass:.4f}", **kw)
    fig.tight_layout()
    finish(fig, "gallery")


FIGURES = {
    "plateau_examples": fig_plateau_examples,
    "dirac_delta_forms": fig_dirac_delta_forms,
    "initial_guess": fig_initial_guess,
    "operators": fig_operators,
    "admm": fig_admm,
    "surface_extraction": fig_surface_extraction,
    "refinement": fig_refinement,
    "gallery": fig_gallery,
}


def main(names):
    ASSETS.mkdir(exist_ok=True)
    for name in names or FIGURES:
        print(f"[{name}]", flush=True)
        FIGURES[name]()


if __name__ == "__main__":
    main(sys.argv[1:])
