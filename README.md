# Computing Minimal Surfaces with Differential Forms

An implementation of [Wang & Chern, *Computing Minimal Surfaces with Differential
Forms*, ACM TOG 40(4), 2021](https://dl.acm.org/doi/10.1145/3450626.3459781):
Plateau's problem for an arbitrary closed curve in 3D, solved as a convex
optimization over differential forms rather than by mesh evolution.

Given a boundary curve Γ, the minimal surface is represented by its Dirac-δ
1-form η on the 3-torus, and found by solving

```
minimize  ‖η‖_mass   subject to   dη = δ_Γ   and a fixed cohomology class
```

Because the constraint set is `η₀ + im(d)`, this reduces to an unconstrained
convex problem in a scalar potential φ, solved by ADMM with Nesterov
acceleration.

## Install

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Animations additionally need manim and its system libraries (cairo, pango,
ffmpeg): `pip install -e ".[animation,notebook]"`.

## Use

```python
from src import curves
from src.plateau import solve_plateau
from src.extract import extract_surface, surface_area

solution = solve_plateau(curves.trefoil, resolution=64)
vertices, faces = extract_surface(solution)
print(solution.mass, surface_area(vertices, faces))
```

`gamma` is any `callable(t) -> (3,)` on `[0, 1)`, or an `(M, 3)` array of
polyline points. `src/curves.py` ships circle, ellipse, polygon, triangle,
trefoil, helicoid and Borromean rings.

For curves that close only up to a lattice translation (`curves.helicoid`), the
projected-area integral is not meaningful — pass the cohomology class explicitly
via `area=`.

## Layout

| Module | Role |
| --- | --- |
| `src/grid.py` | Periodic grid; Fourier symbols of the discrete exterior derivative |
| `src/spectral.py` | DEC operators `d0/d1/d2`, adjoints `delta1/2/3`, Laplacian, mass norm, shrinkage |
| `src/curves.py` | Boundary curve library, projected area vector |
| `src/initial_guess.py` | δ_Γ rasterization, Biot–Savart, cohomology correction (their Algorithm 5) |
| `src/plateau.py` | ADMM solver (their Algorithm 1) |
| `src/extract.py` | Level set + marching cubes → triangle mesh (their §4.1) |

## Validation

The circle is the case with a closed-form answer: its minimal surface is the flat
disc of area πr². For r = 0.3 (exact 0.28274):

| N | `sigma_cells=0` | rel. error | `sigma_cells=1` (default) | rel. error |
| --- | --- | --- | --- | --- |
| 16 | 0.31631 | +11.9% | 0.29611 | +4.7% |
| 32 | 0.30304 | +7.2% | 0.29473 | +4.2% |
| 64 | 0.29500 | +4.3% | 0.29087 | +2.9% |
| 96 | 0.29207 | +3.3% | — | — |

Unmollified, the error is first order in *h* with a clean interpretation: the
excess divided by the rim perimeter is 0.0178, 0.0108, 0.0065, 0.0050 against
*h*/2 = 0.031, 0.016, 0.0078, 0.0052 — the discrete surface is thickened by half
a cell at its boundary. Mollifying δ_Γ over one cell introduces a second error
term of opposite sign that partially cancels this, giving a lower absolute error
at every resolution at the cost of a flatter convergence curve. It is the
default; pass `sigma_cells=0.0` to measure pure discretization error.

Any *planar* curve gives a second class of closed-form checks, since its minimal
surface is just the region it bounds:

| curve | exact | N=32 | N=48 | excess/perimeter ÷ (h/2), N=48 |
| --- | --- | --- | --- | --- |
| circle, r = 0.3 | 0.28274 | +4.2% | +3.4% | 0.49 |
| ellipse, 0.35 × 0.2 | 0.21991 | +4.9% | +3.9% | 0.47 |
| triangle | 0.07500 | +8.9% | +7.0% | 0.40 |

The last column is the same half-cell boundary layer measured on three different
shapes. The triangle's larger *relative* error is entirely its higher
perimeter-to-area ratio, not a different failure mode — corners are handled fine.

`tests/` also checks that the DEC complex is exact (`d∘d = 0`, adjointness,
`Laplacian = dᵀd`), that the initial guess satisfies both constraints, and that
the ADMM fixed point is independent of τ.

## Corrections to the paper

Several details in the published algorithm are either erroneous or unstated. Each
is fixed here with a test guarding it.

**The φ-step's Laplacian does not match its own D.** The paper defines `D` as the
midpoint rule (eq. 23) and states Laplacian `= DᵀD`, but Algorithm 3 supplies the
7-point stencil — a different operator, whose symbols diverge by a factor of 4 at
high frequency. Composing them leaves a **71% relative residual** in what is
supposed to be an exact argmin, which voids ADMM's convergence guarantee.

*Fix:* take `d` to be the **forward** difference. Then `dᵀd` reproduces
Algorithm 3's 7-point stencil to 2e-16 — their Poisson solver was right all along;
it is the midpoint `D` that fails to pair with it. Forward differences also
localize a jump to one cell, which matters because the minimizer is a
discontinuous Dirac-δ form and the mass norm sums `|X|` without letting
oscillations cancel. (A fully spectral `d` is exact too, but rings across the
whole domain and charges ~3.6× for the same jump; it converges to a mass ~62%
above the true area. The midpoint rule additionally has the checkerboard mode in
its kernel, which is presumably why the paper substituted the 7-point stencil in
the first place.)

**eq. (29) uses τλ where it needs λ/τ.** Completing the square on their eq. (28),
`|X| − ⟨λ,X⟩ + (τ/2)|X − W|²`, gives `Z = W + λ/τ`. Algorithm 4 prints
`Z ← τλ̂ + Dφ + X₀`, contradicting their own Algorithm 2, which correctly uses
`λ̂/τ`. The two coincide only at τ = 1 — the default — so the error stays hidden
until τ is tuned. `test_fixed_point_is_independent_of_tau` guards this.

**Algorithm 5 line 21 contradicts eq. (23).** eq. (23) defines the sharp operator
`X = η♯` as a symmetric *average* of the two incident edges; line 21 writes it as
a difference, which is a derivative, not an averaging. Working in collocated
density units throughout makes the conversion the identity, so the ambiguity
disappears.

**The Biot–Savart solve needs the positive semi-definite Laplacian.** Their
Algorithm 3 divides by `+w`, and `ψ = Δ⁺⁻¹δ_Γ` is required for
`d(η₀) = +δ_Γ`. Using the analyst's sign yields `−δ_Γ` — the spanning surface with
reversed orientation, which then fights the `+A` cohomology correction.

**Algorithm 5 lines 12–18 drop a normalization.** They accumulate
`A − Σ_v (η̃₀)_{i,v}` over every vertex and add the result to every vertex, with
no division by `|V|` and no `h²`. The correct statement of eq. (37) in density
units is simply `mean(η₀[..., i]) = A_i`.

**No guidance is given on τ.** The step size cannot be scale-free: the shrinkage
threshold is `1/τ` in absolute field units while `|η₀|` is set by the curve and
the resolution. With τ = 1 and a typical η₀ whose bulk magnitude is well under 1,
the first X-update thresholds the entire field to zero. `_auto_tau` sets τ from
the mean mass density, making it resolution-independent. τ affects only the
convergence rate, not the fixed point.

**A convergence criterion that cannot detect stalling.** Their `c` measures how
far the iterates moved, so it goes to zero when the method stalls just as surely
as when it converges. `solve_plateau` instead stops on the two KKT conditions:
primal feasibility `Dφ − X + X₀ → 0` and dual feasibility `Dᵀλ → 0` (φ carries no
cost, so that is exactly the dual condition). Both are recorded in
`solution.history`, along with `c` for comparison and
`plateau.optimality_residual`, which implements the geometric condition of their
Theorem 1 — at the optimum the normalization `ξ = η/|η|` is coclosed, `δξ = 0`.

Note that `Dᵀλ` is not independent: the φ-step forces
`Dᵀ(Dφ − X̂ + X₀ + λ̂/τ) = 0`, so `Dᵀλ = τDᵀ(X̂ − X)`, the usual ADMM dual
residual. Normalizing it by `‖Dᵀλ‖` therefore divides the quantity by itself; it
is scaled by `h/‖λ‖` instead.

Residuals decay roughly like 1/k, but the mass norm settles to four digits about
an order of magnitude earlier. For area estimates a loose `rtol` is usually
enough.

## Reference

Stephanie Wang and Albert Chern. 2021. Computing Minimal Surfaces with
Differential Forms. *ACM Trans. Graph.* 40, 4, Article 113.
