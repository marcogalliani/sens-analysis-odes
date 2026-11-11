"""
Šesták–Berggren parameter-correlation explorer — self-hosted Streamlit dashboard.

Compares three estimators of the parameter covariance (see docs/param-covariance.md):

  fisher()    — NLS on the exact ODE solution.  H₀(θ) = ½‖z*(θ) − Y‖²
                Σ₀ = σ²(SᵀS)⁻¹, S = ∂z*/∂θ.  A genuine NLS problem (z* does not depend on
                the data), hence the Cramér–Rao bound. No λ: this is the reference line.

  profiling() — generalized profiling.  H(θ) = SSE(x_θ, θ), penalty only in the inner problem.
                Σ₃ = (ZᵀZ)⁻¹ [σ² A Aᵀ] (ZᵀZ)⁻¹,  Z = dz(x_θ, θ)/dθ.
                An M-estimator: x_θ depends on the data. This is what the doc calls
                Strategy 1, and the doc's σ²(J_θᵀJ_θ)⁻¹ for it is WRONG at finite λ —
                correct only as λ → ∞.

  sandwich()  — penalized outer objective (envelope theorem).  H̃(θ) = J(x_θ, θ)
                Σ₂ = B⁻¹ M B⁻¹,  B = J_θθ − J_θx J_xx⁻¹ J_xθ,  M = σ² A Aᵀ.
                The doc's Strategy 2, correct as stated.

Only correlations are reported, and σ² is a common scalar factor in all three, so the noise
level cancels: the knob that separates them is λ.

Two parameterisations of the same model (PARAMETERISATIONS, switchable in the UI):

  "alpha" (default) — conversion-fraction state α ∈ (0, 1) with an affine observation operator

        dα/dt = k(T; Ea, k_ref) · α^m · (1 − α)^n,            α(0) = α0
        y(t)  = (C0 − L0)(1 − α(t)) + L0  =  C0 − (C0 − L0) α(t)

      C0 and L0 appear ONLY in the observation operator; the ODE (and the tracking penalty
      built from it) depends on (k_ref, Ea, m, n, α0) alone.

  "y" — the original measured-quantity state, observed directly

        dy/dt = −k(T; Ea, k_ref) / (C0 − L0)^(m+n−1) · (C0 − y)^m · (y − L0)^n,   y(0) = Y0

  Substituting y = C0 − (C0 − L0) α maps one onto the other exactly (same k), so the two give the
  same trajectories; they differ only in the initial-condition parameter (α0 ↔ Y0 via
  Y0 = C0 − (C0 − L0) α0) and in which variable the tracking penalty is measured in.

Observation model. The state trajectory x lives on a fine STATE grid (n_state nodes); the data Y
lives on a sparse OBSERVATION grid (n_obs ≤ n_state nodes). The predicted data are

        z(x, θ) = O · (a(θ) + c(θ) x)          (per temperature)

with O (n_obs × n_state) the node-selection operator and (a, c) the affine observation map —
(0, 1) for "y", (C0, −(C0 − L0)) for "alpha". SSE(x, θ) = ½‖z(x, θ) − Y‖². Its Gauss–Newton
blocks are

        E_x = ∂z/∂x = c·O,        E_θ = ∂z/∂θ = O (1·∂a/∂θ + x ∂c/∂θ)

so the data-fit contributes E_xᵀE_x to J_xx, E_xᵀE_θ to J_xθ and E_θᵀE_θ to J_θθ. With the "y"
parameterisation E_x = O and E_θ = 0, recovering the earlier formulas (J_xx = OᵀO + λRᵀR etc.).
Setting n_obs = n_state gives O = I.

Each temperature has its OWN operator O_T, so the sampling schedule may differ between
temperatures (and be empty for some); everything above holds per temperature, and the per-T
contributions are summed. In the app the design — which temperatures, and when each one is
sampled — is edited interactively on the setup plot; see init_design() and apply_design_edit().

Two properties the implementation must satisfy, both verified for each parameterisation:
  * λ → ∞: the inner problem is forced onto the ODE manifold and all three coincide. This is the
    sharpest test of the code — see _blocks().
  * Reparametrisation consistency: Σ_alpha(λ·(C0−L0)²) = J Σ_y(λ) Jᵀ for all three estimators,
    J = ∂ln θ_alpha/∂ln θ_y. The penalty in y is (C0−L0)² times the penalty in α, so λ in the
    y-model is λ·(C0−L0)² in the α-model; beyond that factor (and the α0 ↔ Y0 change of IC
    parameter) the two estimators are the same function of θ, hence the same covariance.
"""

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# Runge-Solver implementation
def rk4_solve(rhs_fn, y0, times):
    ns, nv = len(times), len(y0)
    y = np.zeros((ns, nv))
    y[0] = y0
    for i in range(ns - 1):
        dt = times[i + 1] - times[i]
        t = times[i]
        k1 = np.array(rhs_fn(t, y[i]))
        k2 = np.array(rhs_fn(t + dt / 2, y[i] + dt / 2 * k1))
        k3 = np.array(rhs_fn(t + dt / 2, y[i] + dt / 2 * k2))
        k4 = np.array(rhs_fn(t + dt, y[i] + dt * k3))
        y[i + 1] = y[i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y

# Constants
R_GAS = 8.314
CLIP = 1e-15         # relative clip off the rhs roots — tiny, so an IC ε ≥ 1e-8 away from a root is honoured
FD_TH = 1e-4         # default relative (log-parameter) FD step


# Model
def arrhenius(theta, Tk, Tref):
    """k(T) = kref·exp(−Ea/R·(1/T − 1/Tref)).

    The rate is parametrised by kref = k(Tref), the rate at the mean temperature, NOT by the Arrhenius
    pre-exponential A. The two are related by A = kref·exp(Ea/(R·Tref)) — a nonlinear (Ea-dependent)
    reparametrisation, so it changes correlations. A is a huge extrapolation of k to 1/T→0 and is
    near-perfectly confounded with Ea over any realistic temperature window (ρ(A,Ea)→1); centring the
    rate at Tref decouples them, so ρ(kref,Ea) reports the identification the data actually carries.
    This matches the surface driver's make_field."""
    kref, Ea = theta[0], theta[1]
    return kref * np.exp(-Ea / R_GAS * (1 / Tk - 1 / Tref))


def sb_rhs_y(y, theta, Tk, Tref):
    """Original SB rhs in the measured quantity y, which degrades from C0 towards L0.
    Elementwise in y, so it also serves as the collocation vector field on a whole trajectory."""
    _, _, m, n, L0, C0, _ = theta
    k = arrhenius(theta, Tk, Tref)
    yc = np.clip(y, L0 + CLIP * (C0 - L0), C0 - CLIP * (C0 - L0))
    return -k / (C0 - L0) ** (m + n - 1) * (C0 - yc) ** m * (yc - L0) ** n


def sb_rhs_alpha(a, theta, Tk, Tref):
    """SB rhs in the conversion fraction α ∈ (0, 1): dα/dt = k·α^m·(1 − α)^n.
    Independent of C0 and L0 — they live in the observation operator only."""
    m, n = theta[2], theta[3]
    k = arrhenius(theta, Tk, Tref)
    ac = np.clip(a, CLIP, 1 - CLIP)
    return k * ac ** m * (1 - ac) ** n


def _check_y(theta):
    _, _, _, _, L0, C0, Y0 = theta
    if not (L0 < Y0 < C0):
        return (f"Need L0 < Y0 < C0 (got L0={L0:g}, Y0={Y0:g}, C0={C0:g}). The trajectory starts "
                "at y0 = Y0; at y = C0 it hits a root of the SB rhs and at y = L0 it is clipped, so "
                "outside this range the solution degenerates and the sensitivities are unreliable.")
    return None


def _check_alpha(theta):
    _, _, _, _, L0, C0, a0 = theta
    if not L0 < C0:
        return f"Need L0 < C0 (got L0={L0:g}, C0={C0:g})."
    if not 0 < a0 < 1:
        return (f"Need 0 < α0 < 1 (got α0={a0:g}). α = 0 is a root of the SB rhs: for m ≥ 1 the "
                "trajectory never leaves it, and for m < 1 it escapes through a non-Lipschitz point.")
    return None


def _fd_eps_alpha(theta, x0):
    """The α-rhs roots (0, 1) do not move with θ, so a uniform log-step is safe."""
    return np.full(len(theta), FD_TH)


def _fd_eps_y(theta, x0):
    """The y-rhs root y = C0 moves with C0. ∂Φ/∂C0 is taken at FIXED states, the first of which sits
    only C0 − x0 below the root, so the C0 step must not cross it: δC0 ≤ 1e-2·(C0 − x0). Same for a
    free Y0, whose step pushes x0 towards the root. With a free Y0 at the default operating point
    (C0 − Y0 = 0.04) this is still FD_TH; it only shrinks for an IC within ~1e-2 of C0."""
    e = np.full(len(theta), FD_TH)
    d = theta[5] - x0
    e[5] = min(FD_TH, 1e-2 * d / theta[5])
    e[6] = min(FD_TH, 1e-2 * d / abs(theta[6]))
    return e


@dataclass(frozen=True)
class Parameterisation:
    """Everything that differs between the two equivalent SB models, and between IC modes.

    theta always has the layout [kref, Ea, m, n, L0, C0, <IC>]; only the meaning of the last slot
    (Y0, α0 or ε) and the roles of L0, C0 change.

      rhs(x, θ, T, Tref)  state vector field, elementwise in x
      obs(θ) → (a, c)     affine observation map  y = a + c·x
      x0(θ)               initial state
      ic_free             whether theta[6] is an estimable parameter (False: it is a fixed ε)
      xscale(θ)           state range, for the base ∂Φ/∂x FD step
      dist(x, θ)          distance of each state to the nearest rhs root (caps the ∂Φ/∂x FD step)
      fd_eps(θ, x0)       per-parameter log-FD steps, chosen so no perturbation crosses a root
      check(θ)            None if θ is admissible, else a user-facing message
      ic_desc             human-readable IC, for titles
    """
    key: str
    title: str
    pkeys: tuple
    plabel: dict
    rhs: Callable
    obs: Callable
    x0: Callable
    ic_free: bool
    xscale: Callable
    dist: Callable
    fd_eps: Callable
    check: Callable
    ic_desc: str


PARAMETERISATIONS = {
    "alpha": Parameterisation(
        key="alpha",
        title=r"$\dot\alpha = k\,\alpha^m(1-\alpha)^n$,  $y = C_0-(C_0-L_0)\,\alpha$",
        pkeys=("kref", "Ea", "m", "n", "L0", "C0", "alpha0"),
        plabel={"kref": r"$k_\mathrm{ref}$", "Ea": r"$E_a$", "m": "m", "n": "n",
                "L0": r"$L_0$", "C0": r"$C_0$", "alpha0": r"$\alpha_0$"},
        rhs=sb_rhs_alpha,
        obs=lambda th: (th[5], -(th[5] - th[4])),
        x0=lambda th: th[6],
        ic_free=True,
        xscale=lambda th: 1.0,
        dist=lambda x, th: np.minimum(x, 1 - x),
        fd_eps=_fd_eps_alpha,
        check=_check_alpha,
        ic_desc=r"$\alpha_0$ estimated",
    ),
    "y": Parameterisation(
        key="y",
        title=r"$\dot y = -k\,(C_0-y)^m(y-L_0)^n/(C_0-L_0)^{m+n-1}$, $y$ observed",
        pkeys=("kref", "Ea", "m", "n", "L0", "C0", "Y0"),
        plabel={"kref": r"$k_\mathrm{ref}$", "Ea": r"$E_a$", "m": "m", "n": "n",
                "L0": r"$L_0$", "C0": r"$C_0$", "Y0": r"$Y_0$"},
        rhs=sb_rhs_y,
        obs=lambda th: (0.0, 1.0),
        x0=lambda th: th[6],
        ic_free=True,
        xscale=lambda th: th[5] - th[4],
        dist=lambda x, th: np.minimum(th[5] - x, x - th[4]),
        fd_eps=_fd_eps_y,
        check=_check_y,
        ic_desc=r"$Y_0$ estimated",
    ),
}


# Initial-condition modes. "free": theta[6] (α0 or Y0) is an estimable parameter, y(0) = Y0 or
# α(0) = α0. The other two FIX the IC through a small known ε (theta[6] = ε, never estimated) to
# describe a reaction that has just started at t = 0:
#   "eps"         α-model: α(0) = ε               ⇔  y(0) = C0 − (C0 − L0)·ε
#                 y-model: y(0) = C0 − ε          ⇔  α(0) = ε / (C0 − L0)
#   "eps_matched" y-model: y(0) = C0 − (C0 − L0)·ε ⇔  α(0) = ε   — the y-model constraint that
#                 carries EXACTLY the α-model's "eps" information (α-model: same as "eps").
# "eps" in the two models are therefore DIFFERENT constraints: they differ by whether the gap
# from the root scales with C0 − L0. With a free IC the two models are a reparametrisation of
# each other; once the IC is fixed they are only equivalent with the matched y-model constraint.
IC_MODES = ("free", "eps", "eps_matched")


def _check_eps(theta, *, upper):
    _, _, _, _, L0, C0, e = theta
    if not L0 < C0:
        return f"Need L0 < C0 (got L0={L0:g}, C0={C0:g})."
    if not 0 < e < upper(theta):
        return f"Need 0 < ε < {upper(theta):g} (got ε={e:g})."
    return None


def get_par(par_key, ic_mode="free"):
    """The parameterisation `par_key` with the initial-condition mode `ic_mode` (see IC_MODES)."""
    base = PARAMETERISATIONS[par_key]
    if ic_mode == "free":
        return base
    if par_key == "alpha":                                   # "eps" and "eps_matched" coincide
        x0, upper = (lambda th: th[6]), (lambda th: 1.0)
        desc = r"$\alpha_0=\varepsilon$ fixed"
    elif ic_mode == "eps":
        x0, upper = (lambda th: th[5] - th[6]), (lambda th: th[5] - th[4])
        desc = r"$Y_0=C_0-\varepsilon$ fixed"
    elif ic_mode == "eps_matched":
        x0, upper = (lambda th: th[5] - (th[5] - th[4]) * th[6]), (lambda th: 1.0)
        desc = r"$Y_0=C_0-(C_0-L_0)\,\varepsilon$ fixed ($\equiv\alpha_0=\varepsilon$)"
    else:
        raise ValueError(f"unknown IC mode {ic_mode!r}")
    return replace(base, pkeys=base.pkeys[:6] + ("eps",),
                   plabel={**base.plabel, "eps": r"$\varepsilon$"},
                   x0=x0, ic_free=False, ic_desc=desc,
                   check=lambda th: _check_eps(th, upper=upper))

REFINE = 10          # RK4 substeps per state interval (Φ accuracy)
T_H = 8.0            # time horizon of every experiment


def _grids(t_h, n_state):
    """State grid t_fine[idx] (n_state nodes carrying x and the penalty), plus the refined grid
    the ODE is actually integrated on for Φ accuracy."""
    n_fine = (n_state - 1) * REFINE + 1
    t_fine = np.linspace(0.0, t_h, n_fine)
    return t_fine, np.arange(0, n_fine, REFINE)


def uniform_nodes(n_state, n_obs):
    """n_obs state-node indices spread uniformly across the grid, endpoints included (clamped to
    n_state; rounding can merge nodes, so fewer may come back)."""
    n_obs = min(int(n_obs), int(n_state))
    return np.unique(np.linspace(0, n_state - 1, n_obs).round().astype(int))


def _selection(nodes, n_state):
    """O (len(nodes) × n_state): the node-selection operator. Data enters only here."""
    nodes = np.unique(np.clip(np.asarray(nodes, int), 0, n_state - 1))
    O = np.zeros((len(nodes), n_state))
    O[np.arange(len(nodes)), nodes] = 1.0
    return O


def _obs_operator(n_state, n_obs):
    """O (n_obs × n_state) for n_obs uniformly spread nodes. n_obs = n_state gives O = I — the
    fully-observed case."""
    return _selection(uniform_nodes(n_state, n_obs), n_state)


def _obs_operators(n_state, obs, n_temps):
    """One observation operator per temperature.

    obs is either an int — the same n_obs uniform nodes at every temperature — or a sequence with
    one entry per temperature, each a collection of observed state-node indices. The sampling
    schedule may then differ between temperatures, and an entry may be empty: that temperature
    then carries no data and only the penalty sees it (its contribution to every estimator is 0)."""
    if np.isscalar(obs):
        return [_obs_operator(n_state, obs)] * n_temps
    if len(obs) != n_temps:
        raise ValueError(f"obs has {len(obs)} sampling schedules for {n_temps} temperatures")
    return [_selection(nodes, n_state) for nodes in obs]


def _n_samples(obs, n_state, n_temps):
    return sum(O.shape[0] for O in _obs_operators(n_state, obs, n_temps))


def _trajectory(par, theta, Tk, t_fine, idx, Tref):
    rhs = lambda tt, z: [par.rhs(z[0], theta, Tk, Tref)]
    return rk4_solve(rhs, [par.x0(theta)], t_fine)[:, 0][idx]


def _outputs(par, theta, T_K, t_fine, idx, Os):
    """Predicted data z = O_T(a + c·x_T): each per-temperature state trajectory on the state grid,
    mapped through the observation operator, projected to that temperature's observed nodes, then
    stacked. kref is referenced to Tref = mean(T_K)."""
    Tref = float(np.mean(T_K))
    a, c = par.obs(theta)
    return np.concatenate([O @ (a + c * _trajectory(par, theta, Tk, t_fine, idx, Tref))
                           for Tk, O in zip(T_K, Os)])


def _flow_map(par, x, theta, Tk, dt_fine, Tref):
    """Φ: one state interval of RK4, applied elementwise to a vector of states.
    The collocation residual is built from Φ (not from a finite-difference derivative
    operator) so that the physics constraint the penalty encodes is *the same map* the
    Strategy-1 solver uses. Any mismatch here shows up amplified in the correlations,
    because SᵀS is ill-conditioned (cond ~ 1e6)."""
    x = np.asarray(x, float)
    for _ in range(REFINE):
        k1 = par.rhs(x, theta, Tk, Tref)
        k2 = par.rhs(x + dt_fine / 2 * k1, theta, Tk, Tref)
        k3 = par.rhs(x + dt_fine / 2 * k2, theta, Tk, Tref)
        k4 = par.rhs(x + dt_fine * k3, theta, Tk, Tref)
        x = x + dt_fine / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def _setup(temps_C, n_state, t_h):
    T_K = np.asarray(temps_C, float) + 273.15
    t_fine, idx = _grids(t_h, n_state)
    return T_K, t_fine, idx


def _log_fd(fn, theta, cols, eps):
    """Central FD of fn(θ) w.r.t. ln θ_j for j in cols, stacked as the last axis; eps[j] is the
    relative step for parameter j (par.fd_eps — the SAME steps in fisher() and _blocks(), needed
    for the λ→∞ agreement). Log-parameter sensitivities: the parameters differ by many orders of
    magnitude. Correlations are invariant to this diagonal rescaling; cond() is not."""
    out = []
    for j in cols:
        tp, tm = theta.copy(), theta.copy()
        tp[j] *= (1 + eps[j]); tm[j] *= (1 - eps[j])
        out.append((np.asarray(fn(tp), float) - np.asarray(fn(tm), float)) / (2 * eps[j]))
    if not out:
        return np.zeros(np.shape(fn(theta)) + (0,))
    return np.stack(out, axis=-1)


# Strategy 0 — Fisher information of the unpenalized outer objective
def fisher(theta, temps_C, *, par, free=None, n_state=40, obs=40, t_h=8.0):
    """Σ₀ = (SᵀS)⁻¹ (σ² factored out); see _fisher_bm."""
    B, _ = _fisher_bm(theta, temps_C, par=par, free=free, n_state=n_state, obs=obs, t_h=t_h)
    return _safe_inv(B)


def _fisher_bm(theta, temps_C, *, par, free=None, n_state=40, obs=40, t_h=8.0):
    """(B, M) = (SᵀS, SᵀS) with S = ∂z*/∂ln θ — the exact-solution sensitivity of the predicted DATA at the observed nodes,
    so the observation operator's own θ-dependence (C0, L0 in the α-model) and the IC shift are
    both included, and sparse data enters the Cramér–Rao bound honestly. `obs`: see _obs_operators."""
    free = list(par.pkeys if free is None else free)
    theta = np.asarray(theta, float)
    T_K, t_fine, idx = _setup(temps_C, n_state, t_h)
    Os = _obs_operators(n_state, obs, len(T_K))
    cols = [par.pkeys.index(k) for k in free]
    eps = par.fd_eps(theta, par.x0(theta))
    S = _log_fd(lambda th: _outputs(par, th, T_K, t_fine, idx, Os), theta, cols, eps)
    return S.T @ S, S.T @ S            # σ² drops out of the correlation


def _blocks(theta, temps_C, *, par, free=None, n_state=40, obs=40, t_h=8.0):
    """Per-temperature (R, G, E_x, E_θ) shared by both sandwiches.

    Inner residual on the STATE grid:
        r_0 = x_0 − x0(θ)              (initial condition)
        r_i = x_i − Φ(x_{i−1}; θ)      (physics: the same RK4 map fisher() integrates)
    Data residual on the OBSERVATION grid:
        e = O_T(a(θ) + c(θ) x) − Y      (O_T: this temperature's sampling schedule)

    R = ∂r/∂x is unit-diagonal with −∂Φ/∂x on the subdiagonal; G = ∂r/∂θ has first row −∂x0/∂θ.
    E_x = ∂e/∂x = c·O and E_θ = ∂e/∂θ = O(∂a/∂θ + x ∂c/∂θ). Building the penalty from the
    solver's own flow map, and anchoring the initial condition, are both load-bearing: they are
    what make −R⁻¹G reproduce the Strategy-0 state sensitivity, hence what make all three
    strategies agree as λ → ∞. A plain Dx − f(x,θ) collocation residual has no initial condition,
    so its R inherits the kernel of the variational operator and is near-singular.
    """
    free = list(par.pkeys if free is None else free)
    theta = np.asarray(theta, float)
    cols = [par.pkeys.index(k) for k in free]
    T_K, t_fine, idx = _setup(temps_C, n_state, t_h)
    Tref = float(np.mean(T_K))
    Os = _obs_operators(n_state, obs, len(T_K))

    dt_fine = t_fine[1] - t_fine[0]
    I_N = np.eye(n_state)
    i = np.arange(1, n_state)

    eps = par.fd_eps(theta, par.x0(theta))
    # ∂x0/∂θ: with a free IC, nonzero for the IC parameter only. With a fixed-ε IC it is nonzero
    # wherever x0 depends on θ — e.g. y0 = C0 − ε loads ∂y0/∂C0 = 1 onto C0's column.
    dx0dth = _log_fd(par.x0, theta, cols, eps)                   # (P,)
    _, c = par.obs(theta)
    dadth = _log_fd(lambda th: par.obs(th)[0], theta, cols, eps)  # (P,)
    dcdth = _log_fd(lambda th: par.obs(th)[1], theta, cols, eps)  # (P,)

    out = []
    for Tk, O in zip(T_K, Os):
        x = _trajectory(par, theta, Tk, t_fine, idx, Tref)
        xp = x[:-1]                                  # x_{i−1}, the point each step starts from

        # ∂Φ/∂x — Φ acts elementwise, so the state Jacobian of the step is diagonal. The step is
        # capped at 1e-3 × the distance to the nearest root, so an IC ε from the root is resolved.
        hx = np.minimum(1e-6 * par.xscale(theta), 1e-3 * par.dist(xp, theta))
        dphidx = (_flow_map(par, xp + hx, theta, Tk, dt_fine, Tref)
                  - _flow_map(par, xp - hx, theta, Tk, dt_fine, Tref)) / (2 * hx)

        # ∂Φ/∂θ in log-parameters, matching fisher(), restricted to the free parameters
        dphidth = _log_fd(lambda th: _flow_map(par, xp, th, Tk, dt_fine, Tref), theta, cols, eps)

        R = I_N.copy()                               # row 0 is e_0ᵀ: ∂r_0/∂x = e_0
        R[i, i - 1] = -dphidx
        G = np.zeros((n_state, len(cols)))
        G[0, :] = -dx0dth                            # ∂r_0/∂θ = −∂x0/∂θ
        G[i, :] = -dphidth

        Ex = c * O                                   # ∂e/∂x
        Eth = O @ (np.ones((n_state, 1)) * dadth + np.outer(x, dcdth))   # ∂e/∂θ
        out.append((R, G, Ex, Eth))
    return out


# Strategy 2 — sandwich covariance of the penalized outer objective
def sandwich(theta, temps_C, *, par, free=None, n_state=40, obs=40, t_h=8.0, lam=1.0):
    """Σ₂ = B⁻¹ M B⁻¹; see _sandwich_bm."""
    return _sandwich_cov(*_sandwich_bm(theta, temps_C, par=par, free=free, n_state=n_state,
                                       obs=obs, t_h=t_h, lam=lam))


def _sandwich_cov(B, M):
    Binv = _safe_inv(B)
    Sigma = Binv @ M @ Binv
    return 0.5 * (Sigma + Sigma.T)


def _sandwich_bm(theta, temps_C, *, par, free=None, n_state=40, obs=40, t_h=8.0, lam=1.0):
    """(B, M) of Σ₂ = B⁻¹ M B⁻¹ with Gauss-Newton Hessian blocks of J(x, θ) = SSE(x, θ) + λ PEN(x, θ).

    The penalty is the one-step (shooting) residual of the same RK4 map Strategy 0 integrates,
    with the initial condition as its first row:

        r_0 = x_0 − x0(θ),    r_i = x_i − Φ(x_{i−1}; θ),    PEN = ½‖r‖²

    Blocks are assembled at the true θ and its exact trajectory x_θ, where r = 0 and e = 0 to
    machine precision, so residual-weighted second derivatives drop and Gauss-Newton is exact:

        J_xx = E_xᵀE_x + λRᵀR,   J_xθ = E_xᵀE_θ + λRᵀG,   J_θθ = E_θᵀE_θ + λGᵀG

    Envelope theorem: the score is g = ∂J/∂θ at x̂(θ) = E_θᵀe + λGᵀr, and x̂ responds to the data
    through dx̂/dY = J_xx⁻¹E_xᵀ, so

        B = J_θθ − J_θx J_xx⁻¹ J_xθ,      A = dg/dY = −E_θᵀ + J_θx J_xx⁻¹ E_xᵀ,      M = A Aᵀ

    With E_x = O, E_θ = 0 (the y-model) this is the earlier λGᵀG − λ²UᵀV bread and λ²Vᵀ(OᵀO)V meat.

    λ → ∞ limit: the inner problem is forced onto the ODE manifold, x̂ → x*(θ), and H̃(θ) →
    SSE(x*(θ), θ), the NLS objective — so Σ₂ → Σ₀ regardless of how sparse the data is.
    """
    P = len(free or par.pkeys)
    B = np.zeros((P, P))
    M = np.zeros((P, P))

    for R, G, Ex, Eth in _blocks(theta, temps_C, par=par, free=free, n_state=n_state,
                                 obs=obs, t_h=t_h):
        Jxx = Ex.T @ Ex + lam * (R.T @ R)
        Jxth = Ex.T @ Eth + lam * (R.T @ G)
        Jthth = Eth.T @ Eth + lam * (G.T @ G)
        W = np.linalg.solve(Jxx, Jxth)               # J_xx⁻¹ J_xθ  (= −dx̂/dθ)
        A = -Eth.T + W.T @ Ex.T                      # dg/dY  (J_xx symmetric)

        B += Jthth - Jxth.T @ W
        M += A @ A.T                                 # σ² factored out
    return B, M


# Strategy 3 — generalized profiling: penalty in the inner problem, plain SSE outside
def profiling(theta, temps_C, *, par, free=None, n_state=40, obs=40, t_h=8.0, lam=1.0):
    """Σ₃ = (ZᵀZ)⁻¹ [σ² A Aᵀ] (ZᵀZ)⁻¹; see _profiling_bm."""
    return _sandwich_cov(*_profiling_bm(theta, temps_C, par=par, free=free, n_state=n_state,
                                        obs=obs, t_h=t_h, lam=lam))


def _profiling_bm(theta, temps_C, *, par, free=None, n_state=40, obs=40, t_h=8.0, lam=1.0):
    """(Bread, Meat) of Σ₃ = (ZᵀZ)⁻¹ [σ² A Aᵀ] (ZᵀZ)⁻¹, the sandwich for H(θ) = SSE(x_θ, θ) = ½‖z(x_θ, θ) − Y‖².

    Its outer objective is the plain data-fit of the *smoothed* trajectory at the observed nodes,
    so the penalty acts only through x_θ. Because x_θ depends on the data (dx_θ/dY = J_xx⁻¹E_xᵀ ≠ 0),
    this is an M-estimator, NOT an NLS problem — the naive NLS formula is wrong at finite λ and
    only becomes right as λ → ∞. With the total sensitivity of the fitted data

        Z = dz(x_θ, θ)/dθ = E_θ + E_x X,    X = dx_θ/dθ = −J_xx⁻¹ J_xθ

    the score is g = Zᵀe, the bread is ZᵀZ (Gauss-Newton, e = 0 at the truth with clean data) and
    dg/dY = A = Zᵀ(E_x J_xx⁻¹ E_xᵀ − I_obs). With E_x = O, E_θ = 0 this reduces to the earlier
    Xᵀ(OᵀO)X bread and XᵀOᵀ(O J_xx⁻¹Oᵀ − I) meat.

    As λ → 0 the smoother interpolates the data for ANY θ, so SSE(x_θ) → 0 uniformly and θ
    falls out of the outer objective at leading order — profiling loses identification exactly
    where Strategy 2, which keeps λ·PEN in the outer objective, retains it. The variance is
    correspondingly worse: Σ₀ ⪯ Σ₂ ⪯ Σ₃ in the Loewner order for λ ≳ 0.1.
    """
    P = len(par.pkeys if free is None else free)
    Bread = np.zeros((P, P))
    Meat = np.zeros((P, P))

    for R, G, Ex, Eth in _blocks(theta, temps_C, par=par, free=free, n_state=n_state,
                                 obs=obs, t_h=t_h):
        I_obs = np.eye(Ex.shape[0])
        Jxx_inv = _safe_inv(Ex.T @ Ex + lam * (R.T @ R))
        X = -Jxx_inv @ (Ex.T @ Eth + lam * (R.T @ G))     # dx_θ/dθ = −J_xx⁻¹ J_xθ
        Z = Eth + Ex @ X                                  # dz/dθ, total
        A = Z.T @ (Ex @ Jxx_inv @ Ex.T - I_obs)           # dg/dY

        Bread += Z.T @ Z
        Meat += A @ A.T                                   # σ² factored out
    return Bread, Meat


def _safe_inv(X):
    try:
        return np.linalg.inv(X)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(X)


def _corr(Cov):
    sd = np.sqrt(np.clip(np.diag(Cov), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        return Cov / np.outer(sd, sd)


def theta_from_knobs(m, n, L0, C0, ic, Ea, kref):
    """theta = [kref, Ea, m, n, L0, C0, ic], with ic = α0 / Y0 (free IC) or ε (fixed IC)."""
    return np.array([kref, Ea, m, n, L0, C0, ic])


# ── Rate parameter at a reference conversion ──────────────────────────────────────────────────
# ln(dα/dt) = ln k + m·ln α + n·ln(1−α) is a regression whose "covariates" ln α, ln(1−α) are
# negative everywhere on (0, 1): k is the rate extrapolated to αᵐ = (1−α)ⁿ = 1, which no
# conversion reaches, so k compensates every change of m or n (ρ(k,m), ρ(k,n) → 1) and no sampling
# design can centre the covariates. Expressing the rate at reference conversions inside the data,
#
#     r = k · α_rᵐ · (1 − α_s)ⁿ,      dα/dt = r · (α/α_r)ᵐ · ((1−α)/(1−α_s))ⁿ,
#
# is the same model (same curves, same fit, same Var m, Var n, ρ(m,n)) with the rate parameter
# defined where the data are — exactly like centring k_ref at T̄ for Ea. α_s = α_r is the rate at
# one conversion; two references can zero ρ(r,m) and ρ(r,n) exactly, because the information-
# weighted centre of (ln α, ln(1−α)) lies strictly inside eᵘ + eᵛ < 1, off the curve a single
# conversion can reach. In log-parameters ln k = ln r − m·a − n·b, a = ln α_r, b = ln(1−α_s).
# Every estimator's (B, M) transforms with A = ∂ln θ/∂ln θ' as AᵀBA, AᵀMA (all blocks are built
# from first derivatives), so the reparameterisation is exact for all three — and "fix k_ref"
# then means fixing r, a different constraint, handled by restricting AFTER transforming.

RATE_MODES = {
    "k": "k_ref — rate constant (centred at T̄ only)",
    "single": "r(α_r) — rate at a reference conversion α_r",
    "two": "two references α_r, α_s — exact decorrelation",
}


def _ea_coef(Ea, T_from_C, T_to_C):
    """∂ln k(T_from)/∂ln Ea at fixed k(T_to): −(Ea/R)(1/T_from − 1/T_to)."""
    return -(Ea / R_GAS) * (1 / (T_from_C + 273.15) - 1 / (T_to_C + 273.15))


def rate_jacobian(keys, theta, ref, Tbar_C):
    """A = ∂ln θ/∂ln θ' for θ = (k(T̄), …) → θ' = (r, …), r the rate at the reference
    (α_r, α_s, T_r); ref = (a, b, T_r) with a = ln α_r, b = ln(1−α_s), T_r in °C:
        ln k(T̄) = ln r − m·a − n·b − (Ea/R)(1/T̄ − 1/T_r)."""
    A = np.eye(len(keys))
    if ref is not None and "kref" in keys:
        ik = keys.index("kref")
        if "m" in keys:
            A[ik, keys.index("m")] = -theta[2] * ref[0]
        if "n" in keys:
            A[ik, keys.index("n")] = -theta[3] * ref[1]
        if "Ea" in keys:
            A[ik, keys.index("Ea")] = _ea_coef(theta[1], Tbar_C, ref[2])
    return A


def _restricted_cov(B, M, A, keys, free):
    fi = [keys.index(k) for k in free]
    Bt, Mt = A.T @ B @ A, A.T @ M @ A
    return _sandwich_cov(Bt[np.ix_(fi, fi)], Mt[np.ix_(fi, fi)])


def choose_reference(Sig, free, theta, mode, alpha_r=None, Tbar_C=None):
    """Reference (ln α_r, ln(1−α_s), T_r °C) for the rate parameter, or None for k_ref.

    Sig: covariance of the FREE parameters in the k(T̄) parameterisation. In ∂ln θ'/∂ln θ the
    rate row is ln r = ln k + c_m·ln m + c_n·ln n + c_E·ln Ea (locally), c_m = m·a, c_n = n·b,
    c_E = −(Ea/R)(1/T_r − 1/T̄). The temperature reference is always chosen to zero ρ(r, Ea) — the
    plain mean T̄ only does that for k, not for a rate at another conversion. single: α_s = α_r,
    given (alpha_r) or the value minimising ρ(r,m)² + ρ(r,n)² (with c_E solved for each α_r).
    two: (c_m, c_n, c_E) solve Cov(ln r, ·) = 0 exactly — the regression of ln k on the others."""
    shape = [k for k in ("m", "n") if k in free]
    if mode == "k" or "kref" not in free or not shape:
        return None
    m, n, Ea = theta[2], theta[3], theta[1]
    ik = free.index("kref")
    iE = free.index("Ea") if "Ea" in free else None

    def to_ref(a, b, cE):
        T_r = Tbar_C
        if cE and iE is not None:
            inv = 1 / (Tbar_C + 273.15) - cE * R_GAS / Ea
            T_r = 1 / inv - 273.15 if inv > 0 else np.inf
        return (float(a), float(b), float(T_r))

    def cE_for(cm, cn):                                   # zero Cov(ln r, ln Ea)
        if iE is None:
            return 0.0
        cov = Sig[ik, iE]
        if "m" in free:
            cov += cm * Sig[free.index("m"), iE]
        if "n" in free:
            cov += cn * Sig[free.index("n"), iE]
        return -cov / Sig[iE, iE]

    def corr_with(cm, cn, cE):
        J = np.eye(len(free))
        if "m" in free:
            J[ik, free.index("m")] = cm
        if "n" in free:
            J[ik, free.index("n")] = cn
        if iE is not None:
            J[ik, iE] = cE
        return _corr(J @ Sig @ J.T)

    if mode == "two":
        S = [free.index(k) for k in ("m", "n", "Ea") if k in free]
        c = dict(zip([free[j] for j in S],
                     -np.linalg.solve(Sig[np.ix_(S, S)], Sig[S, ik])))
        a = c["m"] / m if "m" in c else None
        b = c["n"] / n if "n" in c else None
        if a is None:                                     # m fixed: its reference is moot
            a = float(np.log(-np.expm1(b))) if b < 0 else 0.0
        if b is None:
            b = float(np.log(-np.expm1(a))) if a < 0 else 0.0
        return to_ref(a, b, c.get("Ea", 0.0))

    if alpha_r is not None:
        a, b = np.log(alpha_r), np.log1p(-alpha_r)
        return to_ref(a, b, cE_for(m * a, n * b))
    best = None
    for x in 1 / (1 + np.exp(-np.linspace(-8.0, 8.0, 641))):  # logit grid on (0, 1)
        a, b = np.log(x), np.log1p(-x)
        cE = cE_for(m * a, n * b)
        C = corr_with(m * a, n * b, cE)
        v = sum(C[ik, free.index(k)] ** 2 for k in shape)
        if best is None or v < best[0]:
            best = (v, a, b, cE)
    return to_ref(*best[1:])


# Cache the (somewhat) expensive computations across reruns. All args are hashable.
@st.cache_data
def compute(par_key, ic_mode, m, n, L0, C0, ic, Ea, kref, temps, n_state, obs, t_h, lam, free,
            rate_mode="k", alpha_r=None):
    """obs: one tuple of observed state-node indices per temperature (see _obs_operators).
    rate_mode / alpha_r: rate parameter (RATE_MODES; alpha_r None = automatic). Returns the three
    correlation matrices, trΣ/trΣ₀, the free parameters, the reference (ln α_r, ln(1−α_s), T_r)
    used (None for k_ref) and, for automatic references, each estimator's own choice."""
    par = get_par(par_key, ic_mode)
    theta = theta_from_knobs(m, n, L0, C0, ic, Ea, kref)
    temps = np.asarray(temps)
    free = [k for k in free if par.ic_free or k != "eps"]      # ε is never estimated
    keys = [k for k in par.pkeys if k != "eps"]
    kw = dict(par=par, free=keys, n_state=n_state, obs=obs, t_h=t_h)
    BM = [_fisher_bm(theta, temps, **kw),                      # NLS / Cramér-Rao bound (λ-free)
          _profiling_bm(theta, temps, lam=lam, **kw),          # generalized profiling
          _sandwich_bm(theta, temps, lam=lam, **kw)]           # penalized outer
    base = [_restricted_cov(B, M, np.eye(len(keys)), keys, free) for B, M in BM]
    # one reference for all three panels (chosen on NLS, the bound) so they compare the same
    # parameter; each estimator's own automatic choice is returned for comparison — at finite λ
    # the tracking estimators draw on a different part of the curve, so their centre differs.
    Tbar = float(np.mean(temps))
    ref = choose_reference(base[0], free, theta, rate_mode, alpha_r, Tbar)
    own = ([choose_reference(C, free, theta, rate_mode, None, Tbar) for C in base]
           if ref is not None and alpha_r is None else None)
    A = rate_jacobian(keys, theta, ref, Tbar)
    covs = [_restricted_cov(B, M, A, keys, free) for B, M in BM]
    # efficiency loss vs the bound: total variance relative to NLS
    eff = [np.trace(C) / np.trace(covs[0]) for C in covs]
    return [_corr(C) for C in covs], eff, free, ref, own


@st.cache_data
def compute_setup(par_key, ic_mode, m, n, L0, C0, ic, Ea, kref, temps, Tref_C, n_state, t_h):
    """Measured-quantity curves y(t) = a + c·x(t) of the SB solution at the current θ, one per
    temperature in `temps` (included or not), on the fine RK4 grid and on the state grid — the same
    trajectories fisher()/_blocks() use. kref is the rate at Tref_C = T*, so the curves depend on
    the physics only, not on which temperatures are in the design."""
    par = get_par(par_key, ic_mode)
    theta = theta_from_knobs(m, n, L0, C0, ic, Ea, kref)
    T_K, t_fine, idx = _setup(temps, n_state, t_h)
    Tref = float(Tref_C) + 273.15
    a, c = par.obs(theta)
    curves = []
    for Tk in T_K:
        rhs = lambda tt, z, Tk=Tk: [par.rhs(z[0], theta, Tk, Tref)]
        curves.append(a + c * rk4_solve(rhs, [par.x0(theta)], t_fine)[:, 0])
    curves = np.array(curves).reshape(len(T_K), len(t_fine))
    return t_fine, curves, curves[:, idx]


# ── Optimal experimental design ───────────────────────────────────────────────────────────────
# Locally optimal design at the current θ for a FIXED SAMPLE BUDGET: choose which (temperature,
# time) pairs to sample, at most one sample per state node and temperature (the observation model
# selects nodes) and at most `max_temps` distinct temperatures. The information model is the NLS
# Fisher information — additive over samples, M = Σ s sᵀ with s = ∂y(T, t)/∂ln θ — i.e. the
# Cramér–Rao bound, which the tracking estimators approach as λ grows. (Their sandwich covariances
# are not additive over samples, so they are evaluated on the result, not optimised.)
#
# Physics vs parameterisation. The kinetics are fixed by k* = k(T*) at a fixed reference T* and
# Ea, so a design change does not change the system. Estimation re-centres k_ref at the design's
# mean temperature T̄ (see arrhenius): ln k* = ln k_ref − (Ea/R)(1/T* − 1/T̄), so in log-parameters
# the sensitivities map as S_est = S* A(T̄), A = I except A[kref, Ea] = −(Ea/R)(1/T* − 1/T̄).
# det A = 1, so the D-criterion over all parameters is T̄-invariant; A, the decorrelation
# criterion and anything with k_ref held fixed are not, and are evaluated in the estimation
# parameterisation of each candidate design.

OED_CRITERIA = {
    "D": "D-optimal — max det M (smallest joint confidence volume)",
    "A": "A-optimal — min tr Σ (smallest summed relative variances)",
    "decor": "Decorrelation — min mean ρ² over parameter pairs",
}


def _batch_states(par, thetas, Tks, Tref, t_fine, idx):
    """RK4 for a batch of (θ, T) at once — thetas (7, B), Tks (B,) — on the state grid (len(idx), B).
    The same arithmetic as rk4_solve, vectorised over the batch."""
    B = len(Tks)
    x = np.broadcast_to(np.asarray(par.x0(thetas), float), (B,)).copy()
    out = np.empty((len(idx), B))
    out[0] = x
    dt, f, k = t_fine[1] - t_fine[0], par.rhs, 1
    for i in range(1, len(t_fine)):
        k1 = f(x, thetas, Tks, Tref)
        k2 = f(x + dt / 2 * k1, thetas, Tks, Tref)
        k3 = f(x + dt / 2 * k2, thetas, Tks, Tref)
        k4 = f(x + dt * k3, thetas, Tks, Tref)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if k < len(idx) and i == idx[k]:
            out[k] = x
            k += 1
    return out


def design_sensitivities(par, theta, temps_C, Tref_C, n_state, t_h=T_H):
    """S[T, i, :] = ∂y(T, t_i)/∂ln θ for every temperature and every state node, w.r.t. every
    estimable parameter (keys), with k_ref = theta[0] the rate at Tref_C. Central differences with
    the same steps as fisher(); one batched integration for all temperatures and perturbations."""
    theta = np.asarray(theta, float)
    keys = [k for k in par.pkeys if k != "eps"]
    cols = [par.pkeys.index(k) for k in keys]
    eps = par.fd_eps(theta, par.x0(theta))
    T_K = np.asarray(temps_C, float) + 273.15
    t_fine, idx = _grids(t_h, n_state)
    nT, P = len(T_K), len(cols)
    thetas = np.repeat(theta[:, None], nT * P * 2, axis=1).reshape(len(theta), nT, P, 2)
    for p, j in enumerate(cols):
        thetas[j, :, p, 0] *= 1 + eps[j]
        thetas[j, :, p, 1] *= 1 - eps[j]
    thetas = thetas.reshape(len(theta), -1)
    Tks = np.repeat(T_K, P * 2)
    x = _batch_states(par, thetas, Tks, float(Tref_C) + 273.15, t_fine, idx)
    a, c = par.obs(thetas)
    y = (a + c * x).reshape(n_state, nT, P, 2)
    S = (y[..., 0] - y[..., 1]) / (2 * eps[cols])
    return keys, np.transpose(S, (1, 0, 2))               # (nT, n_state, P)


def _estimation_info(Ms, Tbars, keys, free, Ea, Tstar_C, ref=None, mn=None):
    """Base information Ms (B, P, P) → information of the FREE parameters with k_ref re-centred at
    each design's mean temperature Tbars (B,) and, if ref = (ln α_r, ln(1−α_s)) is given, expressed
    as the rate at the reference (α_r, α_s, T_r) instead (see rate_jacobian; mn = (m, n))."""
    Ms = np.asarray(Ms, float)
    if "kref" in keys and ("Ea" in keys or ref is not None):
        ik = keys.index("kref")
        A = np.broadcast_to(np.eye(len(keys)), Ms.shape).copy()
        if "Ea" in keys:                       # k* at T* expressed at T̄ — or at T_r if ref
            T_to = np.asarray(Tbars, float) if ref is None else ref[2]
            A[:, ik, keys.index("Ea")] = _ea_coef(Ea, Tstar_C, T_to)
        if ref is not None:
            if "m" in keys:
                A[:, ik, keys.index("m")] = -mn[0] * ref[0]
            if "n" in keys:
                A[:, ik, keys.index("n")] = -mn[1] * ref[1]
        Ms = np.transpose(A, (0, 2, 1)) @ Ms @ A
    fi = [keys.index(k) for k in free]
    return Ms[:, fi][:, :, fi]


def _criterion(Mf, crit):
    """Value to MINIMISE for a batch of free-parameter information matrices; inf if singular."""
    sign, logdet = np.linalg.slogdet(Mf)
    bad = (sign <= 0) | ~np.isfinite(logdet)
    if crit == "D":
        v = -logdet
    else:
        F = Mf.shape[-1]
        Sig = np.linalg.inv(np.where(bad[:, None, None], np.eye(F), Mf))
        if crit == "A":
            v = np.trace(Sig, axis1=1, axis2=2)
        else:
            sd = np.sqrt(np.clip(np.diagonal(Sig, axis1=1, axis2=2), 1e-300, None))
            R = Sig / (sd[:, :, None] * sd[:, None, :])
            v = (np.sum(R ** 2, axis=(1, 2)) - F) / max(F * (F - 1), 1)
    return np.where(bad, np.inf, v)


def design_metrics(S, temps_C, design, keys, free, Ea, Tstar_C, ref=None, mn=None):
    """Summary of a design [(temperature index, node), ...] under the NLS information."""
    used = sorted({ti for ti, _ in design})
    if not design:
        return None
    s = np.array([S[ti, i] for ti, i in design])
    Tbar = float(np.mean([temps_C[ti] for ti in used]))
    Mf = _estimation_info((s.T @ s)[None], [Tbar], keys, free, Ea, Tstar_C, ref, mn)[0]
    sign, logdet = np.linalg.slogdet(Mf)
    if sign <= 0 or not np.isfinite(logdet):
        return {"samples": len(design), "temps": len(used), "logdet": -np.inf, "trace": np.inf,
                "max_rho": np.nan, "rms_rho": np.nan, "Tbar": Tbar}
    C = _corr(np.linalg.inv(Mf))
    od = C[~np.eye(len(free), dtype=bool)]
    return {"samples": len(design), "temps": len(used), "logdet": logdet,
            "trace": float(np.trace(np.linalg.inv(Mf))), "max_rho": float(np.abs(od).max()),
            "rms_rho": float(np.sqrt(np.mean(od ** 2))), "Tbar": Tbar}


def optimal_design(S, temps_C, *, keys, free, Ea, Tstar_C, budget, max_temps, crit="D",
                   n_starts=6, seed=0, max_passes=50, ref=None, mn=None):
    """Search the candidate set {(T, node)} for the best design with `budget` samples.

    Admissible design: at most one sample per (T, node), at most `max_temps` distinct temperatures.
    Each start is improved by two moves until neither helps:
      * point exchange (Fedorov): swap one design point for the admissible candidate that most
        improves `crit`;
      * temperature swap: move all samples of one temperature to the same nodes of an unused
        candidate temperature — point exchange alone can rarely change WHICH temperatures are
        used once max_temps is reached.
    Starts: a D-greedy one (add the candidate that most increases log det of the slightly
    regularised information — the standard way to a non-singular start) plus n_starts − 1 random
    admissible designs (seeded, so the result is reproducible). The best local optimum is returned,
    as a sorted list of (temperature index, node)."""
    nT, N, P = S.shape
    temps_C = np.asarray(temps_C, float)
    flat = S.reshape(nT * N, P)
    outer = np.einsum("ci,cj->cij", flat, flat)
    cand_T = np.repeat(np.arange(nT), N)
    max_temps = int(max(1, min(max_temps, nT)))
    budget = int(min(budget, max_temps * N))
    delta = 1e-8 * float(np.median(np.einsum("cii->c", outer)) or 1.0)
    Ireg = delta * np.eye(len(free))

    def info(Ms, Tbars, reg=False):
        Mf = _estimation_info(Ms, Tbars, keys, free, Ea, Tstar_C, ref, mn)
        return Mf + Ireg if reg else Mf

    class Design:
        def __init__(self, cands):
            self.chosen = np.zeros(len(flat), bool)
            self.M, self.counts = np.zeros((P, P)), np.zeros(nT, int)
            for c in cands:
                self.add(c)

        def add(self, c):
            self.chosen[c] = True; self.M += outer[c]; self.counts[cand_T[c]] += 1

        def remove(self, c):
            self.chosen[c] = False; self.M -= outer[c]; self.counts[cand_T[c]] -= 1

        def candidates(self, crit_, reg=False):
            """Criterion of design ∪ {c} for every admissible c (inf elsewhere)."""
            in_use = self.counts > 0
            n_used, s_used = in_use.sum(), temps_C[in_use].sum()
            new_T = ~in_use[cand_T]
            Tbar = np.where(new_T, (s_used + temps_C[cand_T]) / (n_used + 1),
                            s_used / max(n_used, 1))
            ok = ~self.chosen & (~new_T | (n_used < max_temps))
            vals = np.full(len(flat), np.inf)
            if ok.any():
                vals[ok] = _criterion(info(self.M[None] + outer[ok], Tbar[ok], reg), crit_)
            return vals

        def value(self):
            used = self.counts > 0
            if not used.any():
                return np.inf
            return float(_criterion(info(self.M[None], [temps_C[used].mean()]), crit)[0])

    def improve(d):
        cur = d.value()
        for _ in range(max_passes):
            improved = False
            for r in np.flatnonzero(d.chosen):                       # point exchange
                d.remove(r)
                vals = d.candidates(crit)
                vals[r] = min(vals[r], cur)                          # putting r back is allowed
                c = int(np.argmin(vals))
                if c != r and vals[c] < cur - 1e-10 * max(1.0, abs(cur)):
                    cur, improved = float(vals[c]), True
                else:
                    c = r
                d.add(c)
            used = np.flatnonzero(d.counts > 0)                      # temperature swap
            free_T = np.flatnonzero(d.counts == 0)
            if len(free_T):
                best = (cur, None, None)
                for u in used:
                    nodes = np.flatnonzero(d.chosen[u * N:(u + 1) * N])
                    Mu = outer[u * N + nodes].sum(0)
                    Mv = outer[free_T[:, None] * N + nodes[None, :]].sum(1)      # (|free_T|, P, P)
                    Tbar = (temps_C[used].sum() - temps_C[u] + temps_C[free_T]) / len(used)
                    vals = _criterion(info(d.M[None] - Mu + Mv, Tbar), crit)
                    k = int(np.argmin(vals))
                    if vals[k] < best[0] - 1e-10 * max(1.0, abs(best[0])):
                        best = (float(vals[k]), u, free_T[k])
                if best[1] is not None:
                    _, u, v = best
                    for i in np.flatnonzero(d.chosen[u * N:(u + 1) * N]):
                        d.remove(u * N + i); d.add(v * N + i)
                    cur, improved = d.value(), True
            if not improved:
                break
        return d, cur

    starts = []
    g = Design([])                                                    # D-greedy
    for _ in range(budget):
        vals = g.candidates("D", reg=True)
        c = int(np.argmin(vals))
        if not np.isfinite(vals[c]):
            break
        g.add(c)
    starts.append(g)
    rng = np.random.default_rng(seed)
    for _ in range(max(0, n_starts - 1)):                             # random admissible starts
        Ts = rng.choice(nT, max_temps, replace=False)
        pts = rng.choice(max_temps * N, budget, replace=False)
        starts.append(Design([int(Ts[p // N]) * N + int(p % N) for p in pts]))

    best_d, best_v = None, np.inf
    for d in starts:
        d, v = improve(d)
        if best_d is None or v < best_v:
            best_d, best_v = d, v
    return sorted((int(cand_T[c]), int(c % N)) for c in np.flatnonzero(best_d.chosen))


# ── Interactive experimental design ───────────────────────────────────────────────────────────
# The design lives in st.session_state and is edited from two places — the sidebar and the setup
# plot (a bidirectional custom component). Both mutate it only through callbacks, which Streamlit
# runs before the script, so every widget is then built from the already-updated state.
#   design_pool  every temperature the app knows about [°C], included in the study or not
#   temps_on     the included ones — also the key of the sidebar multiselect
#   samples      {T: [sampling times]}, kept for excluded temperatures too
# Sampling times are stored as times and snapped to the nearest node of the CURRENT state grid
# when used (the observation operator selects state nodes), so a schedule survives a change of
# the state resolution; two times that snap to the same node count once.
DEFAULT_POOL = tuple(round(float(T), 1) for T in np.linspace(5.0, 37.0, 6))
DEFAULT_N_SAMPLES = 10


def _tkey(T):
    return round(float(T), 1)


def node_times(nodes, n_state, t_h=T_H):
    return [round(int(i) * t_h / (n_state - 1), 9) for i in nodes]


def snap_nodes(times, n_state, t_h=T_H):
    """Sorted state-node indices nearest to `times`, duplicates merged."""
    return sorted({min(max(int(round(t / t_h * (n_state - 1))), 0), n_state - 1) for t in times})


def _uniform_times(n, n_state):
    return node_times(uniform_nodes(n_state, n), n_state)


def _ss_int(key, default):
    """Mirror of a slider value from the previous run, readable inside callbacks."""
    return int(st.session_state.get(key, default))


def init_design():
    ss = st.session_state
    if "design_pool" not in ss:
        ss.design_pool = list(DEFAULT_POOL)
        ss.temps_on = list(DEFAULT_POOL)
        ss.samples = {T: _uniform_times(DEFAULT_N_SAMPLES, 40) for T in DEFAULT_POOL}


def add_temperature(T):
    ss = st.session_state
    T = _tkey(T)
    if T not in ss.design_pool:
        ss.design_pool = sorted(ss.design_pool + [T])
        ss.samples[T] = _uniform_times(_ss_int("_n_default", DEFAULT_N_SAMPLES),
                                       _ss_int("_n_state", 40))
    if T not in ss.temps_on:
        ss.temps_on = sorted(ss.temps_on + [T])


def reset_uniform():
    ss = st.session_state
    n, n_state = _ss_int("_n_default", DEFAULT_N_SAMPLES), _ss_int("_n_state", 40)
    ss.samples = {T: _uniform_times(n, n_state) for T in ss.design_pool}


def apply_design_edit(ev):
    """Apply one edit sent by the setup plot. Payloads (see SETUP_JS):
         toggle_temp {T}          include / exclude a temperature
         add_temp {T}             add a temperature to the pool and include it
         delete_temp {T}          drop a temperature from the pool
         toggle_sample {T, i}     add / remove state node i from T's schedule
         range {Ts, i0, i1, mode} add ("add") or remove ("remove") nodes i0..i1 for every T in Ts
         uniform {Ts, n}          n uniformly spread samples for every T in Ts
         clear {Ts}               empty schedule for every T in Ts
         copy {T}                 copy T's schedule to every included temperature"""
    ss = st.session_state
    if not isinstance(ev, dict):
        return
    op, n_state = ev.get("op"), _ss_int("_n_state", 40)
    Ts = [_tkey(T) for T in ev.get("Ts", []) if _tkey(T) in ss.design_pool]
    T = _tkey(ev["T"]) if "T" in ev else None
    node = lambda t: snap_nodes([t], n_state)[0]

    if op == "add_temp":
        add_temperature(T)
    elif T is not None and T not in ss.design_pool:
        return                                             # stale event for a deleted temperature
    elif op == "toggle_temp":
        ss.temps_on = sorted(set(ss.temps_on) ^ {T})
    elif op == "delete_temp":
        ss.design_pool = [x for x in ss.design_pool if x != T]
        ss.temps_on = [x for x in ss.temps_on if x != T]
        ss.samples.pop(T, None)
    elif op == "toggle_sample":
        i, cur = int(ev["i"]), ss.samples.get(T, [])
        hit = [t for t in cur if node(t) == i]
        ss.samples[T] = sorted([t for t in cur if t not in hit] if hit
                               else cur + node_times([i], n_state))
    elif op == "range":
        lo, hi = sorted((int(ev["i0"]), int(ev["i1"])))
        for T_ in Ts:
            cur = [t for t in ss.samples.get(T_, []) if not lo <= node(t) <= hi]
            if ev.get("mode") == "add":
                cur += node_times(range(lo, hi + 1), n_state)
            ss.samples[T_] = sorted(cur)
    elif op == "uniform":
        for T_ in Ts:
            ss.samples[T_] = _uniform_times(max(1, int(ev["n"])), n_state)
    elif op == "clear":
        for T_ in Ts:
            ss.samples[T_] = []
    elif op == "copy":
        for T_ in ss.temps_on:
            ss.samples[T_] = list(ss.samples.get(T, []))


def _on_setup_edit():
    apply_design_edit(st.session_state["setup_plot"].get("edit"))


SETUP_HTML = '<div class="sp-root"></div>'

SETUP_CSS = r"""
.sp-root { --fg: var(--st-text-color, #31333f); --bg: var(--st-background-color, #ffffff);
  --bd: var(--st-border-color, rgba(49, 51, 63, .2));
  --muted: color-mix(in srgb, var(--fg) 60%, transparent);
  --faint: color-mix(in srgb, var(--fg) 13%, transparent);
  --ghost: color-mix(in srgb, var(--fg) 30%, transparent);
  font-family: var(--st-font, "Source Sans Pro", sans-serif); color: var(--fg); font-size: 14px; }
.bar { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: 0 0 8px; }
.lbl { font-weight: 600; font-size: 13px; margin-right: 2px; }
.muted { color: var(--muted); font-size: 12px; }
.chip { display: inline-flex; align-items: center; gap: 6px; padding: 2px 4px 2px 9px;
  border-radius: 999px; border: 1px solid var(--bd); background: var(--bg); color: var(--fg);
  font: inherit; font-size: 13px; cursor: pointer; }
.chip:hover { border-color: var(--muted); }
.chip.off { color: var(--muted); }
.chip.off .name { text-decoration: line-through; }
.sw { width: 10px; height: 10px; border-radius: 50%; box-sizing: border-box;
  background: var(--c); border: 2px solid var(--c); }
.chip.off .sw { background: transparent; }
.del { display: inline-grid; place-items: center; width: 18px; height: 18px; border-radius: 50%;
  color: var(--muted); line-height: 1; }
.del:hover { background: var(--faint); color: var(--fg); }
button.act, select, input { font: inherit; font-size: 13px; color: var(--fg); background: var(--bg);
  border: 1px solid var(--bd); border-radius: 6px; padding: 2px 8px; height: 26px; box-sizing: border-box; }
button.act { cursor: pointer; }
button.act:hover:not(:disabled) { border-color: var(--muted); }
button:disabled { opacity: .45; cursor: not-allowed; }
input[type=number] { width: 68px; }
.hint { color: var(--muted); font-size: 12px; margin-left: auto; }
.plot { position: relative; }
svg { display: block; user-select: none; touch-action: none; cursor: crosshair; }
svg text { fill: var(--fg); font-size: 11px; font-family: inherit; }
svg text.m { fill: var(--muted); }
.tip { position: absolute; pointer-events: none; background: var(--bg); border: 1px solid var(--bd);
  border-radius: 6px; padding: 4px 8px; font-size: 12px; line-height: 1.35; white-space: nowrap;
  box-shadow: 0 2px 8px rgba(0, 0, 0, .14); z-index: 2; }
.tip .k { color: var(--muted); }
.foot { color: var(--muted); font-size: 12px; margin-top: 2px; }
"""

SETUP_JS = r"""
// Interactive experimental-design plot. Python owns the design: every user action is sent as one
// `edit` trigger (payloads documented in apply_design_edit) and is also applied locally at once,
// so the plot responds before the rerun with the authoritative state arrives.
const LIGHT = ["#86b6ef", "#2a78d6", "#0d366b"];   // ordinal blue, cold → hot (light surface)
const DARK  = ["#184f95", "#3987e5", "#cde2fb"];   // same ramp, re-stepped for a dark surface
const fmtT = T => `${+(+T).toFixed(1)} °C`;
const hex = h => { const n = parseInt(h.slice(1), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; };
function ramp(stops, v) {
  v = Math.min(1, Math.max(0, v));
  const s = v * (stops.length - 1), k = Math.min(stops.length - 2, Math.floor(s)), f = s - k;
  const a = hex(stops[k]), b = hex(stops[k + 1]);
  return `rgb(${a.map((x, j) => Math.round(x + (b[j] - x) * f)).join(",")})`;
}
function isDark(el) {
  const p = document.createElement("span"); p.style.color = "var(--bg)"; el.appendChild(p);
  const m = getComputedStyle(p).color.match(/[\d.]+/g); p.remove();
  if (!m) return false;
  const [r, g, b] = m.map(Number);
  return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255 < 0.5;
}
function niceTicks(lo, hi, n) {
  const raw = (hi - lo) / n, p = Math.pow(10, Math.floor(Math.log10(raw))), r = raw / p;
  const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * p, out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * step; v += step) out.push(+v.toFixed(10));
  return out;
}
const uniformNodes = (N, n) => {
  n = Math.max(1, Math.min(n, N));
  if (n === 1) return [0];
  return [...new Set(Array.from({length: n}, (_, k) => Math.round(k * (N - 1) / (n - 1))))];
};

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const root = parentElement.querySelector(".sp-root");
  if (!root || !data) return;
  const S = root.__s || (root.__s = { focus: "all", nU: null, hover: null, drag: null, ro: null });
  S.d = JSON.parse(JSON.stringify(data));
  if (S.nU == null) S.nU = data.n_default;
  const d = S.d, N = d.n_state, tn = d.t_nodes;
  const onCurves = () => d.curves.filter(c => c.on);
  const curve = T => d.curves.find(c => c.T === T);
  if (S.focus !== "all" && !onCurves().some(c => c.T === S.focus)) S.focus = "all";
  const targets = () => S.focus === "all" ? onCurves().map(c => c.T) : [S.focus];
  const send = p => setTriggerValue("edit", p);
  const setSel = (T, sel) => { const c = curve(T); if (c) c.sel = [...new Set(sel)].sort((a, b) => a - b); };

  const dark = isDark(root);
  const Ts = d.curves.map(c => c.T), tmin = Math.min(...Ts), tmax = Math.max(...Ts);
  const color = T => ramp(dark ? DARK : LIGHT, tmax > tmin ? (T - tmin) / (tmax - tmin) : 1);

  root.innerHTML = `
    <div class="bar" role="toolbar" aria-label="Temperatures">
      <span class="lbl">Temperatures</span>
      ${d.curves.map(c => `
        <button class="chip ${c.on ? "on" : "off"}" data-t="${c.T}" aria-pressed="${c.on}"
                title="${c.on ? "Exclude" : "Include"} ${fmtT(c.T)}">
          <i class="sw" style="--c:${color(c.T)}"></i><span class="name">${fmtT(c.T)}</span>
          <span class="del" data-del="${c.T}" title="Remove ${fmtT(c.T)} from the list">×</span>
        </button>`).join("")}
      <input class="addT" type="number" step="0.5" placeholder="°C" aria-label="New temperature in °C">
      <button class="act addB">+ Add</button>
    </div>
    <div class="bar" role="toolbar" aria-label="Sampling">
      <span class="lbl">Edit samples of</span>
      <select class="focus" aria-label="Curve whose samples are edited">
        <option value="all">all included curves</option>
        ${onCurves().map(c => `<option value="${c.T}" ${S.focus === c.T ? "selected" : ""}>${fmtT(c.T)}</option>`).join("")}
      </select>
      <button class="act" data-act="uniform">Uniform</button>
      <input class="nU" type="number" min="1" max="${N}" value="${S.nU}" aria-label="Number of uniform samples">
      <span class="muted">pts</span>
      <button class="act" data-act="clear">Clear</button>
      <button class="act" data-act="copy" ${S.focus === "all" ? 'disabled title="Pick one temperature first"' : 'title="Give every included temperature this schedule"'}>Copy to all</button>
      <span class="hint">Click: add / remove a sample · Drag: add a time window · Alt + drag: remove</span>
    </div>
    <div class="plot"><svg></svg><div class="tip" hidden></div></div>
    <div class="foot"></div>`;

  const svg = root.querySelector("svg"), tip = root.querySelector(".tip"), plot = root.querySelector(".plot");

  // ── toolbar ────────────────────────────────────────────────────────────────────────────────
  root.querySelectorAll(".chip").forEach(b => b.onclick = e => {
    const T = +b.dataset.t;
    if (e.target.closest(".del")) {
      d.curves = d.curves.filter(c => c.T !== T);
      send({ op: "delete_temp", T });
    } else {
      const c = curve(T); c.on = !c.on;
      send({ op: "toggle_temp", T });
    }
    b.classList.toggle("on"); b.classList.toggle("off");
    draw();
  });
  const addT = root.querySelector(".addT");
  const add = () => { const T = parseFloat(addT.value); if (Number.isFinite(T)) send({ op: "add_temp", T }); };
  root.querySelector(".addB").onclick = add;
  addT.onkeydown = e => { if (e.key === "Enter") add(); };
  root.querySelector(".focus").onchange = e => {
    S.focus = e.target.value === "all" ? "all" : +e.target.value;
    const copy = root.querySelector('[data-act="copy"]');
    copy.disabled = S.focus === "all";
    draw();
  };
  root.querySelector(".nU").onchange = e => { S.nU = Math.max(1, Math.min(N, parseInt(e.target.value) || 1)); e.target.value = S.nU; };
  root.querySelectorAll("[data-act]").forEach(b => b.onclick = () => {
    const act = b.dataset.act, Ts = targets();
    if (act === "uniform") { Ts.forEach(T => setSel(T, uniformNodes(N, S.nU))); send({ op: "uniform", Ts, n: S.nU }); }
    if (act === "clear")   { Ts.forEach(T => setSel(T, [])); send({ op: "clear", Ts }); }
    if (act === "copy" && S.focus !== "all") {
      const src = curve(S.focus).sel; onCurves().forEach(c => setSel(c.T, src)); send({ op: "copy", T: S.focus });
    }
    draw();
  });

  // ── chart ──────────────────────────────────────────────────────────────────────────────────
  const M = { l: 58, r: 46, t: 10 }, PH = 290, ROW = 16;
  let G = null;                                           // geometry of the last draw

  function draw() {
    const on = onCurves(), W = Math.max(420, plot.clientWidth || 900), pw = W - M.l - M.r;
    const stripTop = M.t + PH + 46, H = stripTop + Math.max(1, on.length) * ROW + 6;
    const ys = d.curves.flatMap(c => c.y);
    let lo = Math.min(d.L0, ...ys), hi = Math.max(d.C0, ...ys);
    const pad = 0.04 * (hi - lo || 1); lo -= pad; hi += pad;
    const X = t => M.l + t / d.t_h * pw, Y = y => M.t + (hi - y) / (hi - lo) * PH;
    G = { on, W, pw, stripTop, H, X, Y };

    const dim = c => S.focus !== "all" && c.T !== S.focus;
    const path = c => "M" + d.t.map((t, i) => `${X(t).toFixed(1)},${Y(c.y[i]).toFixed(1)}`).join("L");
    const yt = niceTicks(lo, hi, 5), xt = niceTicks(0, d.t_h, 8);
    let s = "";
    yt.forEach(v => s += `<line x1="${M.l}" x2="${M.l + pw}" y1="${Y(v)}" y2="${Y(v)}" style="stroke:var(--faint)"/>
                          <text class="m" x="${M.l - 8}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`);
    xt.forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="${M.t + PH}" y2="${M.t + PH + 4}" style="stroke:var(--muted)"/>
                          <text class="m" x="${X(v)}" y="${M.t + PH + 17}" text-anchor="middle">${v}</text>`);
    s += `<line x1="${M.l}" x2="${M.l + pw}" y1="${M.t + PH}" y2="${M.t + PH}" style="stroke:var(--muted)"/>
          <text class="m" x="${M.l + pw}" y="${M.t + PH + 32}" text-anchor="end">time t</text>
          <text class="m" transform="translate(14 ${M.t + PH / 2}) rotate(-90)" text-anchor="middle">measured y(t)</text>`;
    [[d.C0, "C₀"], [d.L0, "L₀"]].forEach(([v, name]) =>
      s += `<line x1="${M.l}" x2="${M.l + pw}" y1="${Y(v)}" y2="${Y(v)}" style="stroke:var(--muted)" stroke-dasharray="4 3"/>
            <text class="m" x="${M.l + pw + 6}" y="${Y(v) + 4}">${name}</text>`);
    d.curves.filter(c => !c.on).forEach(c =>
      s += `<path d="${path(c)}" fill="none" style="stroke:var(--ghost)" stroke-width="1.5" stroke-dasharray="5 4"/>`);
    on.forEach(c =>
      s += `<path d="${path(c)}" fill="none" stroke="${color(c.T)}" stroke-width="2" opacity="${dim(c) ? .22 : 1}"/>`);
    on.forEach(c => c.sel.forEach(i =>
      s += `<circle cx="${X(tn[i])}" cy="${Y(c.yn[i])}" r="4.5" fill="${color(c.T)}" style="stroke:var(--bg)"
                    stroke-width="1.5" opacity="${dim(c) ? .3 : 1}"/>`));

    // sampling-schedule strip: one row per included temperature
    s += `<text x="${M.l}" y="${stripTop - 14}" font-weight="600">Sampling schedule</text>`;
    on.forEach((c, k) => {
      const y = stripTop + k * ROW;
      s += `<text class="m" x="${M.l - 8}" y="${y + 4}" text-anchor="end">${fmtT(c.T)}</text>
            <line x1="${M.l}" x2="${M.l + pw}" y1="${y}" y2="${y}" style="stroke:var(--faint)"/>`;
      tn.forEach(t => s += `<line x1="${X(t)}" x2="${X(t)}" y1="${y - 2}" y2="${y + 2}" style="stroke:var(--faint)"/>`);
      c.sel.forEach(i => s += `<rect x="${X(tn[i]) - 2}" y="${y - 6}" width="4" height="12" rx="1.5"
                                     fill="${color(c.T)}" opacity="${dim(c) ? .3 : 1}"/>`);
    });
    if (!on.length) s += `<text class="m" x="${M.l}" y="${stripTop + 4}">No temperature included — click a chip above.</text>`;
    s += `<g class="hov"></g><rect class="band" visible="false" style="fill:var(--faint)" height="0" width="0"/>`;

    svg.setAttribute("width", W); svg.setAttribute("height", H); svg.innerHTML = s;

    const total = on.reduce((a, c) => a + c.sel.length, 0);
    root.querySelector(".foot").textContent =
      `${total} samples on ${on.length} temperature${on.length === 1 ? "" : "s"} · ` +
      `curves: k_ref = ${(+d.kref).toPrecision(3)} at T* = ${(+d.T_star).toFixed(1)} °C · estimation re-centres k_ref at ` +
      `the design mean T̄ = ${(+d.Tbar).toFixed(1)} °C (k = ${(+d.kref_c).toPrecision(3)}) · ` +
      `sampling times snap to the ${N}-node state grid (Δt = ${(d.t_h / (N - 1)).toFixed(3)}). ` +
      `Dashed grey: excluded temperatures.`;
  }

  // ── hit-testing & pointer interaction ──────────────────────────────────────────────────────
  const xy = e => { const r = svg.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  const nodeAt = mx => Math.min(N - 1, Math.max(0, Math.round((mx - M.l) / G.pw * (N - 1))));
  const rowAt = my => { const k = Math.round((my - G.stripTop) / ROW); return k >= 0 && k < G.on.length ? k : -1; };
  function hit(mx, my) {
    if (!G || mx < M.l - 6 || mx > M.l + G.pw + 6) return null;
    if (my > G.stripTop - ROW / 2 - 2) {
      const k = rowAt(my);
      return k < 0 ? null : { T: G.on[k].T, i: nodeAt(mx) };
    }
    if (my < M.t || my > M.t + PH) return null;
    if (S.focus !== "all") return { T: S.focus, i: nodeAt(mx) };
    const i0 = nodeAt(mx); let best = null;
    G.on.forEach(c => { for (let i = Math.max(0, i0 - 4); i <= Math.min(N - 1, i0 + 4); i++) {
      const dd = Math.hypot(G.X(tn[i]) - mx, G.Y(c.yn[i]) - my);
      if (dd < 22 && (!best || dd < best.dd)) best = { T: c.T, i, dd };
    }});
    return best;
  }
  function showHover(h, mx, my) {
    const g = svg.querySelector(".hov");
    if (!h) { g.innerHTML = ""; tip.hidden = true; return; }
    const c = curve(h.T), k = G.on.indexOf(c), x = G.X(tn[h.i]), sel = c.sel.includes(h.i), col = color(c.T);
    g.innerHTML = `<line x1="${x}" x2="${x}" y1="${M.t}" y2="${M.t + PH}" style="stroke:var(--faint)"/>
      <circle cx="${x}" cy="${G.Y(c.yn[h.i])}" r="8" fill="none" stroke="${col}" stroke-width="2"/>
      <rect x="${x - 4}" y="${G.stripTop + k * ROW - 8}" width="8" height="16" rx="2" fill="none" stroke="${col}" stroke-width="1.5"/>`;
    tip.innerHTML = `<b>${fmtT(c.T)}</b> <span class="k">·</span> t = ${tn[h.i].toFixed(2)} <span class="k">·</span> y = ${c.yn[h.i].toFixed(3)}<br>
                     <span class="k">${sel ? "sampled — click to remove" : "click to add a sample"}</span>`;
    tip.hidden = false;
    const tw = tip.offsetWidth;
    tip.style.left = `${Math.min(mx + 14, G.W - tw - 4)}px`; tip.style.top = `${Math.max(0, my - 46)}px`;
  }
  function band(x0, x1) {
    const r = svg.querySelector(".band"), a = Math.max(M.l, Math.min(x0, x1)), b = Math.min(M.l + G.pw, Math.max(x0, x1));
    r.setAttribute("x", a); r.setAttribute("width", Math.max(0, b - a));
    r.setAttribute("y", M.t); r.setAttribute("height", G.H - M.t);
  }
  svg.onpointerdown = e => {
    if (e.button !== 0) return;
    const [mx, my] = xy(e), k = my > G.stripTop - ROW / 2 - 2 ? rowAt(my) : -1;
    S.drag = { x0: mx, row: k >= 0 ? G.on[k].T : null, moved: false };
    svg.setPointerCapture(e.pointerId);
  };
  svg.onpointermove = e => {
    const [mx, my] = xy(e);
    if (S.drag) {
      if (Math.abs(mx - S.drag.x0) > 5) S.drag.moved = true;
      if (S.drag.moved) { showHover(null); band(S.drag.x0, mx); return; }
    }
    showHover(hit(mx, my), mx, my);
  };
  svg.onpointerleave = () => { if (!S.drag) showHover(null); };
  svg.onpointerup = e => {
    const [mx, my] = xy(e), dr = S.drag; S.drag = null;
    if (!dr || !G) return;
    if (dr.moved) {
      const a = Math.min(dr.x0, mx), b = Math.max(dr.x0, mx);
      const i0 = Math.max(0, Math.ceil((a - M.l) / G.pw * (N - 1))), i1 = Math.min(N - 1, Math.floor((b - M.l) / G.pw * (N - 1)));
      const Ts = dr.row != null ? [dr.row] : targets(), mode = e.altKey ? "remove" : "add";
      if (i0 <= i1 && Ts.length) {
        Ts.forEach(T => { const c = curve(T), inR = i => i >= i0 && i <= i1;
          setSel(T, mode === "add" ? [...c.sel, ...Array.from({length: i1 - i0 + 1}, (_, j) => i0 + j)] : c.sel.filter(i => !inR(i))); });
        send({ op: "range", Ts, i0, i1, mode });
      }
      draw();
      return;
    }
    const h = hit(mx, my);
    if (!h) return;
    const c = curve(h.T);
    setSel(h.T, c.sel.includes(h.i) ? c.sel.filter(i => i !== h.i) : [...c.sel, h.i]);
    send({ op: "toggle_sample", T: h.T, i: h.i });
    draw(); showHover(h, mx, my);
  };

  draw();
  if (S.ro) S.ro.disconnect();
  let lastW = plot.clientWidth;
  S.ro = new ResizeObserver(() => { if (plot.clientWidth !== lastW) { lastW = plot.clientWidth; draw(); } });
  S.ro.observe(plot);
  return () => { if (S.ro) { S.ro.disconnect(); S.ro = null; } };
}
"""

setup_plot = st.components.v2.component("sb_setup_plot", html=SETUP_HTML, css=SETUP_CSS,
                                        js=SETUP_JS)


def setup_plot_data(t_fine, curves, y_nodes, *, pool, on, samples, n_state, C0, L0, T_star,
                    kref, Tbar, kref_c, n_default, t_h=T_H):
    """JSON payload of the setup plot. The fine curve is thinned to ≤ ~400 points for transport."""
    step = max(1, (len(t_fine) - 1) // 400)
    keep = np.unique(np.r_[np.arange(0, len(t_fine), step), len(t_fine) - 1])
    r = lambda a: np.round(np.asarray(a, float), 6).tolist()
    return {
        "t_h": t_h, "n_state": n_state, "t": r(t_fine[keep]),
        "t_nodes": r(np.linspace(0.0, t_h, n_state)),
        "curves": [{"T": T, "on": T in on, "y": r(curves[k][keep]), "yn": r(y_nodes[k]),
                    "sel": snap_nodes(samples.get(T, []), n_state, t_h)}
                   for k, T in enumerate(pool)],
        "C0": C0, "L0": L0, "T_star": T_star, "kref": kref, "Tbar": Tbar, "kref_c": kref_c,
        "n_default": n_default,
    }


def recentre_kref(kref_star, Ea, T_star_C, Tbar_C):
    """k(T̄) from k(T*): the same kinetics, expressed at the design's mean temperature."""
    return kref_star * np.exp(-Ea / R_GAS * (1 / (Tbar_C + 273.15) - 1 / (T_star_C + 273.15)))


@st.cache_data(show_spinner=False)
def _sens_cached(par_key, ic_mode, knobs, temps, T_star, n_state):
    par = get_par(par_key, ic_mode)
    return design_sensitivities(par, theta_from_knobs(*knobs), temps, T_star, n_state)


@st.cache_data(show_spinner=False)
def _oed_cached(par_key, ic_mode, knobs, cand, T_star, n_state, free, budget, max_temps, crit,
                ref=None):
    keys, S = _sens_cached(par_key, ic_mode, knobs, cand, T_star, n_state)
    d = optimal_design(S, cand, keys=keys, free=list(free), Ea=knobs[5], Tstar_C=T_star,
                       budget=budget, max_temps=max_temps, crit=crit, ref=ref, mn=knobs[:2])
    return [(float(cand[ti]), int(i)) for ti, i in d]


def _metrics_for(par_key, ic_mode, knobs, T_star, n_state, free, design_TN, ref=None):
    """design_metrics of a design given as [(temperature °C, node), ...]."""
    if not design_TN:
        return None
    temps = tuple(sorted({T for T, _ in design_TN}))
    keys, S = _sens_cached(par_key, ic_mode, knobs, temps, T_star, n_state)
    k = {T: j for j, T in enumerate(temps)}
    return design_metrics(S, temps, [(k[T], i) for T, i in design_TN], keys, list(free),
                          knobs[5], T_star, ref, knobs[:2])


def apply_oed(design_TN, n_state):
    """Make the optimal design the study design; the previous one stays undoable."""
    ss = st.session_state
    ss._design_undo = {"pool": list(ss.design_pool), "on": list(ss.temps_on),
                       "samples": {T: list(v) for T, v in ss.samples.items()}}
    temps = sorted({_tkey(T) for T, _ in design_TN})
    ss.design_pool = sorted(set(ss.design_pool) | set(temps))
    ss.temps_on = temps
    for T in temps:
        ss.samples[T] = node_times(sorted(i for T2, i in design_TN if _tkey(T2) == T), n_state)


def undo_oed():
    ss = st.session_state
    u = ss.pop("_design_undo", None)
    if u:
        ss.design_pool, ss.temps_on, ss.samples = u["pool"], u["on"], u["samples"]


def oed_panel(*, par, par_key, ic_mode, knobs, T_star, free, n_state, on_data, obs_all,
              ref=None):
    """Optimal-design tool: pick temperatures and sampling times for a fixed sample budget."""
    import pandas as pd
    ss = st.session_state
    cur = [(T, i) for T in on_data for i in obs_all[T]]
    sig = (par_key, ic_mode, tuple(knobs), T_star, n_state, tuple(free))
    defaults = {"oed_crit": "D", "oed_budget": max(len(cur), 2 * len(free)),
                "oed_maxT": max(len(on_data), 2), "oed_source": "range",
                "oed_Tlo": 5.0, "oed_Thi": 45.0, "oed_Tstep": 2.5}
    for k, v in defaults.items():
        if k not in ss:
            ss[k] = v

    with st.expander("**Optimal design** — choose temperatures and sampling times for a fixed "
                     "sample budget", expanded="oed" in ss):
        st.caption("Locally optimal at the current parameters, with the NLS Fisher information — "
                   "the Cramér–Rao bound, which both tracking estimators approach as λ grows. "
                   "Each temperature/state-node pair can be sampled at most once, and the free "
                   "parameters are the ones not held fixed in the sidebar. Applying the result "
                   "makes it the study design; earlier temperatures stay in the list, excluded.")
        with st.form("oed_form", border=False):
            c1, c2, c3 = st.columns([2.4, 1, 1])
            c1.selectbox("Criterion", list(OED_CRITERIA), format_func=OED_CRITERIA.get,
                         key="oed_crit",
                         help="D: maximise det M, the information about all free parameters "
                              "jointly — invariant to how they are parameterised. A: minimise "
                              "the summed variances of the log-parameters (relative errors). "
                              "Decorrelation: minimise the mean squared correlation; it ignores "
                              "precision, so check tr Σ in the comparison below.")
            c2.number_input("Sample budget (total)", min_value=1, max_value=5000, step=1,
                            key="oed_budget")
            c3.number_input("Max temperatures", min_value=1, max_value=100, step=1,
                            key="oed_maxT",
                            help="Upper bound on the number of temperatures (separate "
                                 "experiments) the design may use.")
            c4, c5, c6, c7 = st.columns([2.4, 1, 1, 1], vertical_alignment="bottom")
            c4.radio("Candidate temperatures", ["range", "current"], horizontal=True,
                     key="oed_source",
                     format_func={"range": "a range",
                                  "current": "the included ones (times only)"}.get)
            c5.number_input("from [°C]", step=1.0, key="oed_Tlo")
            c6.number_input("to [°C]", step=1.0, key="oed_Thi")
            c7.number_input("step [°C]", min_value=0.1, step=0.5, key="oed_Tstep")
            run = st.form_submit_button("Compute optimal design", type="primary")

        if run:
            if ss.oed_source == "current":
                cand = tuple(sorted(ss.temps_on))
            else:
                lo, hi = sorted((ss.oed_Tlo, ss.oed_Thi))
                cand = tuple(float(T) for T in
                             np.unique(np.round(np.arange(lo, hi + 1e-9, ss.oed_Tstep), 1)))
            err = None
            if len(free) < 2:
                err = "Leave at least two parameters free."
            elif not cand:
                err = "No candidate temperatures."
            elif len(cand) > 41:
                err = f"{len(cand)} candidate temperatures — use a coarser step (at most 41)."
            elif "Ea" in free and min(ss.oed_maxT, len(cand)) < 2:
                err = "Ea needs at least two temperatures — raise Max temperatures or fix Ea."
            elif ss.oed_budget < len(free):
                err = f"A budget of {ss.oed_budget} cannot identify {len(free)} free parameters."
            if err:
                st.warning(err)
            else:
                with st.spinner("Searching designs…"):
                    design = _oed_cached(par_key, ic_mode, tuple(knobs), cand, T_star, n_state,
                                         tuple(free), int(ss.oed_budget), int(ss.oed_maxT),
                                         ss.oed_crit, ref)
                ss.oed = {"sig": sig, "design": design, "crit": ss.oed_crit}

        res = ss.get("oed")
        if res and res["sig"] != sig:
            st.info("The operating point, state grid or free parameters changed since the last "
                    "optimal design was computed — compute it again.")
        elif res:
            m_opt = _metrics_for(par_key, ic_mode, knobs, T_star, n_state, free, res["design"],
                                 ref)
            m_cur = _metrics_for(par_key, ic_mode, knobs, T_star, n_state, free, cur, ref)
            F = len(free)

            def row(name, mm):
                if mm is None:
                    return {"design": name}
                return {"design": name, "samples": mm["samples"], "temperatures": mm["temps"],
                        "T̄ [°C]": round(mm["Tbar"], 1),
                        "D-efficiency": float(np.exp((mm["logdet"] - m_opt["logdet"]) / F)),
                        "tr Σ ÷ optimal": mm["trace"] / m_opt["trace"],
                        "max |ρ|": mm["max_rho"], "rms ρ": mm["rms_rho"]}

            st.markdown(f"**Result** — {OED_CRITERIA[res['crit']]}")
            st.dataframe(pd.DataFrame([row("current", m_cur), row("optimal", m_opt)]),
                         hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.3f")
                                        for c in ("D-efficiency", "tr Σ ÷ optimal",
                                                  "max |ρ|", "rms ρ")})
            st.caption("D-efficiency = (det M / det M_optimal)^(1/p): the fraction of the "
                       "optimal design's samples that would carry the same information. "
                       "tr Σ ÷ optimal: summed relative variances, NLS, relative to the optimal "
                       "design. Both use the free parameters with k_ref re-centred at each "
                       "design's own T̄ — and, when the rate is expressed at a reference "
                       "conversion, at the current α_r (D is unaffected by either).")
            t_nodes = np.linspace(0.0, T_H, n_state)
            sched = [{"T [°C]": T, "samples": len(nodes),
                      "sampling times": ", ".join(f"{t_nodes[i]:.2f}" for i in nodes)}
                     for T in sorted({T for T, _ in res["design"]})
                     for nodes in [sorted(i for T2, i in res["design"] if T2 == T)]]
            st.dataframe(pd.DataFrame(sched), hide_index=True, width="stretch")
            b1, b2, _ = st.columns([1, 1, 3])
            b1.button("Apply to the setup", type="primary", on_click=apply_oed,
                      args=(res["design"], n_state), width="stretch")
            b2.button("Undo apply", on_click=undo_oed, disabled="_design_undo" not in ss,
                      width="stretch")


def _heatmap(ax, Corr, labels, title):
    k = len(labels)
    im = ax.imshow(Corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(k)); ax.set_xticklabels(labels)
    ax.set_yticks(range(k)); ax.set_yticklabels(labels)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{Corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(Corr[i, j]) > 0.5 else "black")
    ax.set_title(title, fontsize=8.5)
    return im


def _strongest(Corr, free):
    od = Corr - np.eye(len(free))
    a, b = np.unravel_index(np.argmax(np.abs(od)), od.shape)
    return f"ρ({free[a]},{free[b]}) = {od[a, b]:+.3f}"


def make_figure(corrs, eff, free, *, par, m, n, L0, C0, ic, Ea, kref, Tbar, temps, n_state,
                n_samples, lam, ref=None):
    plabel = dict(par.plabel)
    names = list(free)
    if ref is not None:                                  # rate at the reference conversion(s)
        plabel["kref"] = r"$r$"
        names = ["r" if k == "kref" else k for k in free]
    labels = [plabel[k] for k in free]
    titles = [
        "NLS on the exact ODE solution\n"
        r"$\Sigma_0=\sigma^2(S^TS)^{-1}$ — Cramér–Rao bound, no $\lambda$",
        "Generalized profiling  $H=\\mathrm{SSE}(x_\\theta)$\n"
        rf"M-estimator sandwich, $\lambda$={lam:g}",
        "Penalized outer  $\\tilde{H}=J(x_\\theta,\\theta)$\n"
        rf"envelope-theorem sandwich, $\lambda$={lam:g}",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    fig.subplots_adjust(wspace=0.32, top=0.78)
    for ax, Corr, title, e in zip(axes, corrs, titles, eff):
        im = _heatmap(ax, Corr, labels,
                      f"{title}\nstrongest {_strongest(Corr, names)}\n"
                      f"tr$\\Sigma$/tr$\\Sigma_0$ = {e:.2f}×")
    fig.colorbar(im, ax=list(axes), shrink=0.85)

    fixed = [x for x in par.pkeys if x not in free and x != "eps"]
    ic_name = par.pkeys[6]
    fig.suptitle(f"{par.title},  {par.ic_desc}\n"
                 f"m={m:g}  n={n:g}  L0={L0:g}  C0={C0:g}  {ic_name}={ic:g}  Ea={Ea/1e3:g} kJ/mol  "
                 + (f"k_ref={kref:.3g} at T̄={Tbar:.1f}°C" if ref is None else
                    f"r = rate at α_r={np.exp(ref[0]):.3g}"
                    + (f", α_s={-np.expm1(ref[1]):.3g}" if abs(np.exp(ref[0]) + np.exp(ref[1]) - 1)
                       > 1e-9 else "") + f", T_r={ref[2]:.1f}°C")
                 + f"   |   {len(temps)} temps, {n_samples} samples on {n_state}-node state grid"
                 f"   |   estimated: {len(free)} params "
                 f"(fixed: {'none' if not fixed else ', '.join(fixed)})",
                 fontsize=10)
    return fig


# Streamlit UI
def bounded_slider(label, lo, hi, default, step, *, edit=False, help=None, is_int=False):
    """A slider whose min/max the user can override. When `edit` is on, two number_inputs for the
    bounds appear above it; the value is always clamped into the (possibly rescaled) range, so
    changing the bounds never raises. Keys are derived from the label, which is unique per slider."""
    cast = int if is_int else float
    fmt = "%d" if is_int else None
    if edit:
        c1, c2 = st.columns(2)
        lo = c1.number_input(f"{label} min", value=cast(lo), step=cast(step), format=fmt,
                             key=f"{label}__lo")
        hi = c2.number_input(f"{label} max", value=cast(hi), step=cast(step), format=fmt,
                             key=f"{label}__hi")
    lo, hi, step = cast(lo), cast(hi), cast(step)
    if hi <= lo:                                     # keep a valid, non-empty range
        hi = lo + step
    default = cast(min(max(default, lo), hi))
    return st.slider(label, lo, hi, default, step, format=fmt, help=help)


def _main():
    st.set_page_config(page_title="SB correlation explorer", layout="wide")
    st.title("Parameter-correlation explorer — Šesták–Berggren")
    st.caption("Three estimators of the same parameters, side by side. **NLS** fits the exact ODE "
               "solution and is the Cramér–Rao bound (no λ). **Generalized profiling** and the "
               "**penalized outer** objective are both tracking estimators built on the same "
               "smoothed inner problem, and both are M-estimators whose covariance is a sandwich — "
               "profiling is *not* an NLS problem at finite λ, contrary to the note in "
               "`docs/param-covariance.md`. Both converge to NLS as λ → ∞. "
               "Move the operating point and design knobs; tick a parameter to hold it known. "
               "Build the experimental design on the setup plot: include or exclude temperatures "
               "and click the curves to choose when each one is sampled.")
    init_design()
    ss = st.session_state

    with st.sidebar:
        edit_ranges = st.checkbox("✎ edit slider ranges", value=False,
                                  help="Reveal a min/max input pair above every slider so you can "
                                       "rescale the exploration range. Changing a range simply "
                                       "clamps that slider's value into the new bounds.")
        par_key = st.radio(
            "Parameterisation", list(PARAMETERISATIONS),
            format_func=lambda k: {"alpha": "α-space + observation operator",
                                   "y": "y-space (original)"}[k],
            help="α-space: dα/dt = k·α^m·(1−α)^n with y = C0 − (C0−L0)·α, so C0 and L0 enter only "
                 "the observation operator and the IC parameter is α0. y-space: the original "
                 "dy/dt = −k/(C0−L0)^(m+n−1)·(C0−y)^m·(y−L0)^n with IC parameter Y0. Same "
                 "trajectories; Y0 = C0 − (C0−L0)·α0. The y-space penalty is (C0−L0)² times the "
                 "α-space one, so λ_y corresponds to λ_α = λ_y·(C0−L0)².")
        ic_choice = st.radio(
            "Initial condition", ["free", "eps"], index=1,
            format_func=lambda k: {"free": "estimated (α0 / Y0 is a parameter)",
                                   "eps": "reaction just started (fixed by ε)"}[k],
            help="Estimated: the IC is a free parameter (α0 or Y0). Just started: the IC is pinned "
                 "a known small ε away from the root of the rhs and is NOT estimated — α(0) = ε in "
                 "α-space, y(0) = C0 − (C0−L0)·ε in y-space (the same constraint in both).")
        ic_mode, eps_caption = "free", None
        if ic_choice == "eps":
            # y-space uses the matched constraint, so both parameterisations carry the same IC
            # information (get_par still offers the unmatched y(0) = C0 − ε as ic_mode="eps").
            ic_mode = "eps_matched" if par_key == "y" else "eps"
            log_eps = bounded_slider("log₁₀ ε", -6.0, -1.0, -3.0, 0.25, edit=edit_ranges,
                                     help="Known, fixed gap of the IC from the root of the rhs "
                                          "(α = 0 ⇔ y = C0), as a conversion fraction. Not "
                                          "estimated.")
            ic = 10.0 ** log_eps
            eps_caption = st.empty()                 # filled once C0 and L0 are known
        par = get_par(par_key, ic_mode)

        st.header("Operating point")
        m   = bounded_slider("m",  0.20, 2.00, 0.70, 0.05, edit=edit_ranges)
        n   = bounded_slider("n",  0.20, 2.50, 1.30, 0.05, edit=edit_ranges)
        L0  = bounded_slider("L0", 0.02, 0.50, 0.20, 0.02, edit=edit_ranges)
        if par_key == "alpha":
            C0 = bounded_slider("C0", 0.70, 1.50, 1.00, 0.02, edit=edit_ranges,
                                help="Non-degraded value: y = C0 at α = 0. Enters only the "
                                     "observation operator y = C0 − (C0−L0)·α.")
        else:
            C0 = bounded_slider("C0", 0.70, 1.50, 1.00, 0.02, edit=edit_ranges,
                                help="Non-degraded value the sample would reach with no "
                                     "degradation. It is NOT directly observed — the trajectory "
                                     "starts below C0.")
        D = C0 - L0
        if ic_mode != "free":
            if D > 0:
                eps_caption.caption(f"ε = {ic:.3g}:  α0 = ε  ⇔  Y0 = C0 − (C0−L0)·ε "
                                    f"= {C0 - D * ic:.6g}")
        elif par_key == "alpha":
            ic = bounded_slider("α0", 0.005, 0.50, 0.05, 0.005, edit=edit_ranges,
                                help="Initial conversion fraction α(0), an independent parameter "
                                     "in (0, 1). α = 0 is a root of the SB rhs, so α0 must stay "
                                     "positive. Equivalent measured initial value: "
                                     "Y0 = C0 − (C0−L0)·α0.")
            st.caption(f"⇔ Y0 = C0 − (C0−L0)·α0 = {C0 - D * ic:.4g}")
        else:
            ic = bounded_slider("Y0", 0.30, 1.40, 0.96, 0.02, edit=edit_ranges,
                                help="Measured initial value: y0 = Y0, an independent parameter "
                                     "held strictly between L0 and C0. y = C0 is a root of the SB "
                                     "rhs, so Y0 must stay below C0 (and above L0) or the "
                                     "trajectory degenerates.")
            if D > 0:
                st.caption(f"⇔ α0 = (C0−Y0)/(C0−L0) = {(C0 - ic) / D:.4g}")
        Ea_kJ = bounded_slider("Ea  [kJ/mol]", 20.0, 90.0, 50.0, 2.0, edit=edit_ranges)
        kref  = bounded_slider("k_ref", 0.50, 6.00, 2.50, 0.25, edit=edit_ranges,
                               help="Rate constant at the reference temperature T* below. With Ea "
                                    "it fixes the kinetics at every temperature, so changing the "
                                    "design never changes the system.")
        T_star = st.number_input("T*  — temperature of k_ref [°C]", value=21.0, step=0.5,
                                 key="T_star",
                                 help="Where the k_ref slider is specified. It sets the physics "
                                      "only: for the correlations k_ref is re-centred at the mean "
                                      "temperature of the design (the temperatures that carry "
                                      "data), k(T̄) = k_ref·exp(−Ea/R·(1/T̄ − 1/T*)), which "
                                      "decouples it from Ea. The default is the mean of the "
                                      "default temperatures.")

        st.header("Rate parameter")
        rate_mode = st.radio(
            "Estimate the rate as", list(RATE_MODES), format_func=RATE_MODES.get, key="rate_mode",
            help="ln(dα/dt) = ln k + m·ln α + n·ln(1−α): k is the rate extrapolated to "
                 "αᵐ = (1−α)ⁿ = 1, which no conversion reaches, so it compensates every change of "
                 "m or n. The same model written with the rate at a reference conversion, "
                 "dα/dt = r·(α/α_r)ᵐ·((1−α)/(1−α_s))ⁿ with r = k·α_rᵐ·(1−α_s)ⁿ, defines the rate "
                 "where the data are — like centring k_ref at T̄ for Ea. Curves, fit and the "
                 "uncertainty of m and n are unchanged; only the rate parameter's meaning (and its "
                 "correlations) change. Two references (α_s ≠ α_r) can zero ρ(r,m) and ρ(r,n) "
                 "exactly, at the cost of r no longer being the rate at one conversion.")
        alpha_r = None
        if rate_mode == "single":
            if not st.checkbox("choose α_r automatically", value=True, key="alpha_r_auto",
                               help="The α_r that minimises ρ(r,m)² + ρ(r,n)² for the NLS "
                                    "covariance of the current design; used for all three "
                                    "panels."):
                alpha_r = bounded_slider("α_r", 0.005, 0.995, 0.05, 0.005, edit=edit_ranges,
                                         help="Reference conversion: r is dα/dt at α = α_r "
                                              "(and T = T̄).")

        st.header("Experimental design")
        n_state = bounded_slider("state nodes / temperature", 10, 80, 40, 5, edit=edit_ranges,
                                 is_int=True,
                                 help="Resolution of the trajectory x and the ODE penalty — the "
                                      "state discretization, independent of how much data you "
                                      "have. This is the grid the inner problem lives on.")
        ss._n_state = n_state                    # mirrored for the design callbacks
        st.multiselect("Included temperatures", options=ss.design_pool, key="temps_on",
                       format_func=lambda T: f"{T:g} °C",
                       help="Temperatures in the study. Same as the chips above the setup plot: "
                            "excluded temperatures keep their sampling schedule and stay in the "
                            "list (dashed grey on the plot) so they can be added back.")
        c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
        c1.number_input("New temperature [°C]", value=45.0, step=0.5, key="new_T")
        c2.button("Add", on_click=lambda: add_temperature(ss.new_T), width="stretch")
        n_default = bounded_slider("samples / temperature (uniform)", 1, 80, DEFAULT_N_SAMPLES, 1,
                                   edit=edit_ranges, is_int=True,
                                   help="Uniform schedule given to newly added temperatures and "
                                        "applied by the reset button. Clamped to the number of "
                                        "state nodes. Fine-tune individual schedules on the plot.")
        ss._n_default = n_default
        st.button("Reset all schedules to uniform", on_click=reset_uniform, width="stretch",
                  help="Give every temperature (included or not) the uniform schedule above. "
                       "Data enter only through SSE = ½Σ_T‖O_T(a + c·x_T) − Y_T‖², so the "
                       "data-fit Hessian block of temperature T is c²·O_TᵀO_T (nonzero only at "
                       "its sampled nodes).")

        st.header("Tracking penalty")
        log_lam = bounded_slider("log₁₀ λ", -3.0, 6.0, 0.0, 0.25, edit=edit_ranges,
                                 help="Weight of the ODE penalty in the inner objective. The NLS "
                                      "panel does not depend on it. As λ → ∞ the inner problem is "
                                      "forced onto the ODE manifold and both tracking estimators "
                                      "converge to NLS, so all three panels must coincide; push "
                                      "the slider right to check.")
        lam = 10.0 ** log_lam
        st.caption(f"λ = {lam:.3g}. σ² is a common factor in all three covariances and cancels out "
                   "of the correlations, so it is not a knob here. Beyond λ ≈ 10³ the three panels "
                   "agree to ~1e-4. The trΣ/trΣ₀ figure on each panel is the efficiency loss "
                   "against the Cramér–Rao bound — watch it blow up for profiling as λ → 0, where "
                   "the smoother fits the data for any θ and identification is lost.")

        st.header("Fix (hold known)")
        fixed = {k: st.checkbox(f"fix {k}") for k in par.pkeys if k != "eps"}

    free = tuple(k for k in par.pkeys if k != "eps" and not fixed[k])
    pool, on = list(ss.design_pool), sorted(ss.temps_on)
    Ea = Ea_kJ * 1e3
    msg = par.check(theta_from_knobs(m, n, L0, C0, ic, Ea, kref))

    if msg:
        st.warning(msg)
        return

    obs_all = {T: tuple(snap_nodes(ss.samples.get(T, []), n_state)) for T in on}
    on_data = [T for T in on if obs_all[T]]                 # temperatures that carry data
    Tbar = float(np.mean(on_data)) if on_data else float(T_star)
    kref_c = recentre_kref(kref, Ea, T_star, Tbar)

    st.subheader("Experimental setup")
    if pool:
        t_fine, curves, y_nodes = compute_setup(par_key, ic_mode, m, n, L0, C0, ic, Ea, kref,
                                                tuple(pool), float(T_star), n_state, T_H)
        setup_plot(key="setup_plot", on_edit_change=_on_setup_edit,
                   data=setup_plot_data(t_fine, curves, y_nodes, pool=pool, on=on,
                                        samples=ss.samples, n_state=n_state, C0=C0, L0=L0,
                                        T_star=float(T_star), kref=kref, Tbar=Tbar,
                                        kref_c=kref_c, n_default=n_default))
    else:
        st.info("No temperatures in the study — add one in the sidebar.")

    oed_slot = st.container()                 # filled below, once the rate reference is known
    ref = None
    if rate_mode == "single" and alpha_r is not None and "kref" in free \
            and ({"m", "n"} & set(free)):
        ref = (float(np.log(alpha_r)), float(np.log1p(-alpha_r)), Tbar)

    obs = tuple(obs_all[T] for T in on_data)
    n_samples = sum(map(len, obs))
    st.subheader("Parameter correlations")
    if not on:
        st.warning("Include at least one temperature.")
    elif len(free) < 2:
        st.warning("Leave at least two parameters free to form a correlation matrix.")
    elif n_samples < len(free):
        st.warning(f"{n_samples} samples cannot identify {len(free)} free parameters — add "
                   "samples on the plot or fix some parameters.")
    elif "Ea" in free and len(on_data) < 2:
        st.warning("Ea is only identifiable with data at two or more temperatures — sample "
                   "another temperature or fix Ea.")
    else:
        if rate_mode != "k" and "kref" not in free:
            st.info("k_ref is held fixed, so there is no rate parameter to re-express — showing "
                    "the k_ref parameterisation.")
        elif rate_mode != "k" and not ({"m", "n"} & set(free)):
            st.info("m and n are both fixed, so the reference conversion has no effect.")
        corrs, eff, free_list, ref, own = compute(par_key, ic_mode, m, n, L0, C0, ic, Ea, kref_c,
                                                  tuple(on_data), n_state, obs, T_H, lam, free,
                                                  rate_mode, alpha_r)
        fig = make_figure(corrs, eff, free_list, par=par,
                          m=m, n=n, L0=L0, C0=C0, ic=ic, Ea=Ea, kref=kref_c, Tbar=Tbar,
                          temps=on_data, n_state=n_state, n_samples=n_samples, lam=lam, ref=ref)
        st.pyplot(fig)
        if ref is not None:
            a_r, a_s, T_r = np.exp(ref[0]), -np.expm1(ref[1]), ref[2]
            r_val = recentre_kref(kref, Ea, float(T_star), T_r) * np.exp(m * ref[0] + n * ref[1])
            how = ("chosen automatically on the NLS covariance" if alpha_r is None
                   else "α_r set by hand")
            if rate_mode == "two":
                what = (f"r = k(T_r)·α_rᵐ·(1−α_s)ⁿ = {r_val:.3g} with α_r = {a_r:.3g}, "
                        f"α_s = {a_s:.3g}, T_r = {T_r:.1f} °C ({how}) — a rate scale, no longer "
                        "the rate at a single conversion")
            else:
                what = (f"r = dα/dt at α_r = {a_r:.3g} and T_r = {T_r:.1f} °C = {r_val:.3g} "
                        f"({how})")
            msg = (f"Rate parameter: {what}. T_r is chosen so that ρ(r, Ea) = 0 for NLS — the "
                   f"design mean T̄ = {Tbar:.1f} °C only does that for k. k(T̄) = {kref_c:.3g} "
                   "for comparison.")
            if own is not None:
                fmt = (lambda rf: (f"α_r = {np.exp(rf[0]):.3g}" if rate_mode == "single" else
                                   f"α_r = {np.exp(rf[0]):.3g}, α_s = {-np.expm1(rf[1]):.3g}")
                       + f", T_r = {rf[2]:.1f} °C")
                msg += (" Each estimator's own choice — NLS: " + fmt(own[0])
                        + " · profiling: " + fmt(own[1]) + " · penalized: " + fmt(own[2])
                        + ". At finite λ the tracking estimators draw on a different part of "
                          "the curve, so the NLS reference does not decorrelate them as well; "
                          "they coincide as λ → ∞.")
            st.caption(msg)

    with oed_slot:
        oed_panel(par=par, par_key=par_key, ic_mode=ic_mode, knobs=(m, n, L0, C0, ic, Ea, kref),
                  T_star=float(T_star), free=free, n_state=n_state, on_data=on_data,
                  obs_all=obs_all, ref=ref)


if __name__ == "__main__":
    _main()
