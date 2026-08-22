"""
Šesták–Berggren parameter-correlation explorer — self-hosted Streamlit dashboard.

Compares three estimators of the parameter covariance (see docs/param-covariance.md):

  fisher()    — NLS on the exact ODE solution.  H₀(θ) = ½‖x*(θ) − Y‖²
                Σ₀ = σ²(SᵀS)⁻¹, S = ∂x*/∂θ.  A genuine NLS problem (x* does not depend on
                the data), hence the Cramér–Rao bound. No λ: this is the reference line.

  profiling() — generalized profiling.  H(θ) = SSE(x_θ), penalty only in the inner problem.
                Σ₃ = (XᵀX)⁻¹ [σ² Xᵀ(J_xx⁻¹ − I)² X] (XᵀX)⁻¹,  X = dx_θ/dθ.
                An M-estimator: x_θ depends on the data. This is what the doc calls
                Strategy 1, and the doc's σ²(J_θᵀJ_θ)⁻¹ for it is WRONG at finite λ —
                correct only as λ → ∞.

  sandwich()  — penalized outer objective (envelope theorem).  H̃(θ) = J(x_θ, θ)
                Σ₂ = B⁻¹ M B⁻¹,  B = J_θθ − J_θx J_xx⁻¹ J_xθ,  M = σ² J_θx J_xx⁻² J_xθ.
                The doc's Strategy 2, correct as stated.

Only correlations are reported, and σ² is a common scalar factor in all three, so the noise
level cancels: the knob that separates them is λ.

Data sparsity. The state trajectory x lives on a fine STATE grid (n_state nodes); the data Y
lives on a sparse OBSERVATION grid (n_obs ≤ n_state nodes), tied together by a selection
operator O (n_obs × n_state), so SSE(x) = ½‖O x − Y‖². The data-fit Hessian block is therefore
OᵀO — diagonal, 1 at observed nodes and 0 at unobserved ones — not the identity. Setting
n_obs = n_state (O = I) recovers the fully-observed case. Keeping the two grids separate lets
you thin the data without coarsening the ODE representation; it changes three places:
  * J_xx = OᵀO + λRᵀR                          (was I + λRᵀR)
  * Strategy 2 meat  λ² Vᵀ(OᵀO)V               (was λ² VᵀV)
  * Strategy 1 bread Xᵀ(OᵀO)X, and dx_θ/dY = J_xx⁻¹Oᵀ enters its meat.

Two properties the implementation must satisfy, both verified:
  * λ → ∞: the inner problem is forced onto the ODE manifold and all three coincide
    (agreement ~5e-5). This is the sharpest test of the code — see _blocks().
  * Σ₀ ⪯ Σ₂ ⪯ Σ₃ in the Loewner order for λ ≳ 0.1. Profiling is the LEAST efficient of the
    three at finite λ, because as λ → 0 the smoother fits the data for any θ and the outer
    objective goes flat. Both sandwiches were validated by Monte Carlo against the real
    nonlinear inner solve (bread matches the analytic Hessian to 1e-6).
"""

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
R_GAS, EPS = 8.314, 1e-6
# Parameters
PKEYS  = ["kref", "Ea", "m", "n", "L0", "C0", "Y0"]
PLABEL = {"kref": r"$k_\mathrm{ref}$", "Ea": r"$E_a$", "m": "m", "n": "n", "L0": r"$L_0$", "C0": r"$C_0$", "Y0": r"$Y_0$"}


# Model
def sb_rhs(y, theta, Tk, Tref):
    """SB right-hand side f(y, θ; T). Elementwise in y, so it also serves as the collocation vector
    field on a whole trajectory.

    The rate is parametrised by kref = k(Tref), the rate at the mean temperature, NOT by the Arrhenius
    pre-exponential A. The two are related by A = kref·exp(Ea/(R·Tref)) — a nonlinear (Ea-dependent)
    reparametrisation, so it changes correlations. A is a huge extrapolation of k to 1/T→0 and is
    near-perfectly confounded with Ea over any realistic temperature window (ρ(A,Ea)→1); centring the
    rate at Tref decouples them, so ρ(kref,Ea) reports the identification the data actually carries.
    This matches the surface driver's make_field: k(T) = kref·exp(−Ea/R·(1/T − 1/Tref))."""
    kref, Ea, m, n, L0, C0, Y0 = theta
    k = kref * np.exp(-Ea / R_GAS * (1 / Tk - 1 / Tref))
    yc = np.clip(y, L0 + EPS, C0 - EPS)
    return -k / (C0 - L0) ** (m + n - 1) * (C0 - yc) ** m * (yc - L0) ** n


REFINE = 10          # RK4 substeps per state interval (Φ accuracy)


def _grids(t_h, n_state):
    """State grid t_fine[idx] (n_state nodes carrying x and the penalty), plus the refined grid
    the ODE is actually integrated on for Φ accuracy."""
    n_fine = (n_state - 1) * REFINE + 1
    t_fine = np.linspace(0.0, t_h, n_fine)
    return t_fine, np.arange(0, n_fine, REFINE)


def _obs_operator(n_state, n_obs):
    """O (n_obs × n_state): selects which state nodes are observed. Data enters only here.

    n_obs is clamped to n_state; the observed nodes are spread uniformly across the state grid
    (endpoints included). n_obs = n_state gives O = I — the fully-observed case the earlier
    version of this code assumed everywhere."""
    n_obs = min(int(n_obs), int(n_state))
    obs_nodes = np.unique(np.linspace(0, n_state - 1, n_obs).round().astype(int))
    O = np.zeros((len(obs_nodes), n_state))
    O[np.arange(len(obs_nodes)), obs_nodes] = 1.0
    return O


def _y0(theta):
    """Initial state y0 = Y0, an independent parameter (theta[6]).

        y0 = Y0,   with L0 < Y0 < C0

    Y0 is the measured initial value; C0 is the non-degraded value the sample would reach with no
    degradation and is NOT directly observed. Because y0 is now parametrised through Y0 alone —
    ∂y0/∂lnY0 = Y0, and 0 for every other parameter — the initial-condition sensitivity loads onto
    Y0 (the first row of G in the sandwich, the IC shift in the Strategy-1 sensitivity), decoupling
    it from C0. Previously y0 = C0 − δ tied the start to C0 and suppressed most of C0's own
    influence on the data; splitting Y0 out removes that confound.

    Y0 must stay strictly below C0. y = C0 is a root of the SB rhs: for m ≥ 1 it is a stationary
    point the trajectory never leaves (everything degenerates to NaN), and for m < 1 the solution
    escapes but through a non-Lipschitz point where ∂f/∂y ∝ (C0 − y)^(m−1) is unbounded — so the
    sandwich's state Hessian does not exist there. Keeping Y0 < C0 (and Y0 > L0) holds the start
    off that point; ∂Φ/∂x at the first node scales like (C0 − Y0)^(m−1)."""
    return theta[6]


def _trajectory(theta, Tk, t_fine, idx, Tref):
    rhs = lambda tt, z: [sb_rhs(z[0], theta, Tk, Tref)]
    return rk4_solve(rhs, [_y0(theta)], t_fine)[:, 0][idx]


def _outputs(theta, T_K, t_fine, idx, O):
    """Predicted data: each per-temperature trajectory on the state grid, projected to the
    observed nodes by O, then stacked. kref is referenced to Tref = mean(T_K)."""
    Tref = float(np.mean(T_K))
    return np.concatenate([O @ _trajectory(theta, Tk, t_fine, idx, Tref) for Tk in T_K])


def _flow_map(y, theta, Tk, dt_fine, Tref):
    """Φ: one state interval of RK4, applied elementwise to a vector of states.
    The collocation residual is built from Φ (not from a finite-difference derivative
    operator) so that the physics constraint the penalty encodes is *the same map* the
    Strategy-1 solver uses. Any mismatch here shows up amplified in the correlations,
    because SᵀS is ill-conditioned (cond ~ 1e6)."""
    y = np.asarray(y, float)
    for _ in range(REFINE):
        k1 = sb_rhs(y, theta, Tk, Tref)
        k2 = sb_rhs(y + dt_fine / 2 * k1, theta, Tk, Tref)
        k3 = sb_rhs(y + dt_fine / 2 * k2, theta, Tk, Tref)
        k4 = sb_rhs(y + dt_fine * k3, theta, Tk, Tref)
        y = y + dt_fine / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def _setup(theta, temps_C, n_state, t_h):
    T_K = np.asarray(temps_C, float) + 273.15
    t_fine, idx = _grids(t_h, n_state)
    return T_K, t_fine, idx


# Strategy 1 — Fisher information of the unpenalized outer objective
def fisher(theta, temps_C, *, free=None, n_state=40, n_obs=40, t_h=8.0):
    free = list(PKEYS if free is None else free)
    theta = np.asarray(theta, float)
    T_K, t_fine, idx = _setup(theta, temps_C, n_state, t_h)
    O = _obs_operator(n_state, n_obs)                    # data lives on the observed nodes only

    # log-parameter sensitivities ∂/∂ln θ — the parameters differ by many orders of
    # magnitude. Correlations are invariant to this diagonal rescaling; cond() is not.
    # _outputs recomputes y0(θ) = Y0 from the perturbed θ, so S includes the initial-condition
    # shift through Y0. S = O·∂x*/∂θ — the exact-solution sensitivity AT the observed
    # nodes, so sparse data enters the Cramér–Rao bound honestly.
    eps = 1e-4
    P_all = len(PKEYS)
    n_obs_eff = O.shape[0]
    S = np.zeros((n_obs_eff * len(T_K), P_all))
    for j in range(P_all):
        tp, tm = theta.copy(), theta.copy()
        tp[j] *= (1 + eps); tm[j] *= (1 - eps)
        S[:, j] = (_outputs(tp, T_K, t_fine, idx, O)
                   - _outputs(tm, T_K, t_fine, idx, O)) / (2 * eps)

    Sf = S[:, [PKEYS.index(k) for k in free]]
    return _safe_inv(Sf.T @ Sf)          # σ² drops out of the correlation


def _blocks(theta, temps_C, *, free=None, n_state=40, t_h=8.0):
    """Per-temperature (R, G) of the inner residual on the STATE grid, shared by both sandwiches.

        r_0 = x_0 − y0(θ)              (initial condition)
        r_i = x_i − Φ(x_{i−1}; θ)      (physics: the same RK4 map fisher() integrates)

    R = ∂r/∂x is unit-diagonal with −∂Φ/∂x on the subdiagonal; G = ∂r/∂θ has first row −∂y0/∂θ
    (nonzero, since y0 = Y0 is parametrised through Y0). Building the penalty from the
    solver's own flow map, and anchoring the initial condition, are both load-bearing: they are
    what make −R⁻¹G reproduce the Strategy-0 sensitivity, hence what make all three strategies
    agree as λ → ∞. A plain Dx − f(x,θ) collocation residual has no initial condition, so its R
    inherits the kernel of the variational operator and is near-singular.
    """
    free = list(PKEYS if free is None else free)
    theta = np.asarray(theta, float)
    cols = [PKEYS.index(k) for k in free]
    T_K, t_fine, idx = _setup(theta, temps_C, n_state, t_h)
    Tref = float(np.mean(T_K))

    dt_fine = t_fine[1] - t_fine[0]
    I_N = np.eye(n_state)
    i = np.arange(1, n_state)
    eps = 1e-4

    # ∂y0/∂lnθ — nonzero for Y0 only, since y0 = Y0 (∂y0/∂lnY0 = Y0, zero for the rest)
    dy0dth = np.zeros(len(cols))
    for c, j in enumerate(cols):
        tp, tm = theta.copy(), theta.copy()
        tp[j] *= (1 + eps); tm[j] *= (1 - eps)
        dy0dth[c] = (_y0(tp) - _y0(tm)) / (2 * eps)

    out = []
    for Tk in T_K:
        x = _trajectory(theta, Tk, t_fine, idx, Tref)
        xp = x[:-1]                                  # x_{i−1}, the point each step starts from

        # ∂Φ/∂x — Φ acts elementwise, so the state Jacobian of the step is diagonal
        h = 1e-6 * (theta[5] - theta[4])
        dphidx = (_flow_map(xp + h, theta, Tk, dt_fine, Tref)
                  - _flow_map(xp - h, theta, Tk, dt_fine, Tref)) / (2 * h)

        # ∂Φ/∂θ in log-parameters, matching fisher(), restricted to the free parameters
        dphidth = np.zeros((n_state - 1, len(cols)))
        for c, j in enumerate(cols):
            tp, tm = theta.copy(), theta.copy()
            tp[j] *= (1 + eps); tm[j] *= (1 - eps)
            dphidth[:, c] = (_flow_map(xp, tp, Tk, dt_fine, Tref)
                             - _flow_map(xp, tm, Tk, dt_fine, Tref)) / (2 * eps)

        R = I_N.copy()                               # row 0 is e_0ᵀ: ∂r_0/∂x = e_0
        R[i, i - 1] = -dphidx
        G = np.zeros((n_state, len(cols)))
        G[0, :] = -dy0dth                            # ∂r_0/∂θ = −∂y0/∂θ
        G[i, :] = -dphidth
        out.append((R, G))
    return out


# Strategy 2 — sandwich covariance of the penalized outer objective
def sandwich(theta, temps_C, *, free=None, n_state=40, n_obs=40, t_h=8.0, lam=1.0):
    """Σ₂ = B⁻¹ M B⁻¹ with Gauss-Newton Hessian blocks of J(x, θ) = SSE(x) + λ PEN(x, θ).

    The penalty is the one-step (shooting) residual of the same RK4 map Strategy 1 integrates,
    with the initial condition as its first row:

        r_0 = x_0 − y0(θ),    r_i = x_i − Φ(x_{i−1}; θ),    PEN = ½‖r‖²

    so R = ∂r/∂x is unit-diagonal with −∂Φ/∂x on the subdiagonal, and G = ∂r/∂θ has first row
    −∂y0/∂θ — nonzero, because y0 = Y0 is an estimated parameter. Strategy 1 carries the same
    initial-condition shift.

    Two properties this buys, both of which fail for a plain Dx − f(x,θ) collocation residual:
      * R is well conditioned. The derivative-operator residual has no initial condition, so
        its R inherits the kernel of the variational operator (d/dt − ∂f/∂x) — the homogeneous
        solution — and is near-singular; −R⁻¹G is then swamped by that mode.
      * −R⁻¹G reproduces the Strategy-1 sensitivity to ~1e-5, which makes the two strategies
        provably agree as λ → ∞ (see below).

    Blocks are assembled at the true θ and its exact trajectory x_θ, where r = 0 to machine
    precision, so the r-weighted second derivatives drop and Gauss-Newton is exact.

    Sparse data enters through OᵀO: J_xx = OᵀO + λRᵀR, and the noise reaches θ only through the
    observed nodes, so the meat is λ²Vᵀ(OᵀO)V (V = J_xx⁻¹J_xθ/λ). With O = I this is the
    fully-observed case.

    λ → ∞ limit: with K = (OᵀO + λRRᵀ)⁻¹ ≈ (1/λ)(RRᵀ)⁻¹ for large λ, both bread and meat tend to
    Gᵀ(RRᵀ)⁻¹G = SᵀS with S = −R⁻¹G, hence Σ₂ → σ²(SᵀS)⁻¹ = Σ₀ — the OᵀO term is negligible
    against λRᵀR there, so the three coincide regardless of how sparse the data is.
    """
    P = len(free or PKEYS)
    B = np.zeros((P, P))
    M = np.zeros((P, P))
    OtO = _obs_operator(n_state, n_obs).T @ _obs_operator(n_state, n_obs)  # ∂²SSE/∂x∂xᵀ

    for R, G in _blocks(theta, temps_C, free=free, n_state=n_state, t_h=t_h):
        Jxx = OtO + lam * (R.T @ R)                  # = OᵀO + λ ∂²PEN/∂x∂xᵀ
        U = R.T @ G                                  # J_xθ = λ U
        V = np.linalg.solve(Jxx, U)                  # J_xx⁻¹ J_xθ / λ

        B += lam * (G.T @ G) - lam ** 2 * (U.T @ V)
        M += lam ** 2 * (V.T @ (OtO @ V))            # λ² Vᵀ OᵀO V — noise via observed nodes

    Binv = _safe_inv(B)
    Sigma = Binv @ M @ Binv
    return 0.5 * (Sigma + Sigma.T)


# Strategy 3 — generalized profiling: penalty in the inner problem, plain SSE outside
def profiling(theta, temps_C, *, free=None, n_state=40, n_obs=40, t_h=8.0, lam=1.0):
    """Σ₃ = (XᵀOᵀOX)⁻¹ [σ² Aᵀ… ] (XᵀOᵀOX)⁻¹, the sandwich for H(θ) = SSE(x_θ) = ½‖O x_θ − Y‖².

    Its outer objective is the plain data-fit of the *smoothed* trajectory at the observed nodes,
    so the penalty acts only through x_θ. Because x_θ depends on the data (dx_θ/dY = J_xx⁻¹Oᵀ ≠ 0),
    this is an M-estimator, NOT an NLS problem — the naive NLS formula is wrong at finite λ and
    only becomes right as λ → ∞. The score is

        g(θ) = Xᵀ Oᵀ(O x_θ − Y),   X = dx_θ/dθ = −J_xx⁻¹ J_xθ,   dg/dY = Xᵀ Oᵀ(O J_xx⁻¹Oᵀ − I_obs)

    giving bread Xᵀ(OᵀO)X (Gauss-Newton: O x_θ − Y = 0 at the truth with clean data) and meat AAᵀ.
    With O = I these reduce to XᵀX and Xᵀ(J_xx⁻¹ − I).

    As λ → 0 the smoother interpolates the data for ANY θ, so SSE(x_θ) → 0 uniformly and θ
    falls out of the outer objective at leading order — profiling loses identification exactly
    where Strategy 2, which keeps λ·PEN in the outer objective, retains it. The variance is
    correspondingly worse: Σ₀ ⪯ Σ₂ ⪯ Σ₃ in the Loewner order for λ ≳ 0.1.
    """
    P = len(PKEYS if free is None else free)
    Bread = np.zeros((P, P))
    Meat = np.zeros((P, P))
    O = _obs_operator(n_state, n_obs)
    OtO = O.T @ O
    I_obs = np.eye(O.shape[0])

    for R, G in _blocks(theta, temps_C, free=free, n_state=n_state, t_h=t_h):
        Jxx_inv = _safe_inv(OtO + lam * (R.T @ R))
        X = -lam * (Jxx_inv @ (R.T @ G))             # = dx_θ/dθ = −J_xx⁻¹ J_xθ
        A = X.T @ O.T @ (O @ Jxx_inv @ O.T - I_obs)  # = dg/dY

        Bread += X.T @ (OtO @ X)
        Meat += A @ A.T                              # σ² factored out

    Bi = _safe_inv(Bread)
    Sigma = Bi @ Meat @ Bi
    return 0.5 * (Sigma + Sigma.T)


def _safe_inv(X):
    try:
        return np.linalg.inv(X)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(X)


def _corr(Cov):
    sd = np.sqrt(np.clip(np.diag(Cov), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        return Cov / np.outer(sd, sd)


def theta_from_knobs(m, n, L0, C0, Y0, Ea, kref, temps_C):
    # kref (rate at the mean temperature) is now the estimated rate parameter itself — no conversion
    # to the Arrhenius pre-exponential A. temps_C is unused, kept for a stable call signature.
    return np.array([kref, Ea, m, n, L0, C0, Y0])


# Cache the (somewhat) expensive computations across reruns. All args are hashable.
@st.cache_data
def compute(m, n, L0, C0, Y0, Ea, kref, temps, n_state, n_obs, t_h, lam, free):
    theta = theta_from_knobs(m, n, L0, C0, Y0, Ea, kref, np.asarray(temps))
    temps = np.asarray(temps)
    free = list(free)
    kw = dict(free=free, n_state=n_state, n_obs=n_obs, t_h=t_h)
    C0_ = fisher(theta, temps, **kw)                     # NLS / Cramér-Rao bound (λ-free)
    C3 = profiling(theta, temps, lam=lam, **kw)          # generalized profiling
    C2 = sandwich(theta, temps, lam=lam, **kw)           # penalized outer
    # efficiency loss vs the bound: total variance relative to NLS
    eff = [np.trace(C) / np.trace(C0_) for C in (C0_, C3, C2)]
    return [_corr(C) for C in (C0_, C3, C2)], eff, free


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


def make_figure(corrs, eff, free, *, m, n, L0, C0, Y0, Ea, kref, temps, n_state, n_obs, lam):
    labels = [PLABEL[k] for k in free]
    titles = [
        "NLS on the exact ODE solution\n"
        r"$\Sigma_0=\sigma^2(S^TS)^{-1}$ — Cramér–Rao bound, no $\lambda$",
        "Generalized profiling  $H=\\mathrm{SSE}(x_\\theta)$\n"
        rf"M-estimator sandwich, $\lambda$={lam:g}",
        "Penalized outer  $\\tilde{H}=J(x_\\theta,\\theta)$\n"
        rf"envelope-theorem sandwich, $\lambda$={lam:g}",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    fig.subplots_adjust(wspace=0.32)
    for ax, Corr, title, e in zip(axes, corrs, titles, eff):
        im = _heatmap(ax, Corr, labels,
                      f"{title}\nstrongest {_strongest(Corr, free)}\n"
                      f"tr$\\Sigma$/tr$\\Sigma_0$ = {e:.2f}×")
    fig.colorbar(im, ax=list(axes), shrink=0.85)

    fixed = [x for x in PKEYS if x not in free]
    fig.suptitle(f"m={m:g}  n={n:g}  L0={L0:g}  C0={C0:g}  Y0={Y0:g}  Ea={Ea/1e3:g} kJ/mol  k_ref={kref:g}   "
                 f"|   {len(temps)} temps, {n_obs} obs on {n_state}-node state grid   |   "
                 f"estimated: {len(free)} params (fixed: {'none' if not fixed else ', '.join(fixed)})",
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


st.set_page_config(page_title="SB correlation explorer", layout="wide")
st.title("Parameter-correlation explorer — Šesták–Berggren")
st.caption("Three estimators of the same parameters, side by side. **NLS** fits the exact ODE "
           "solution and is the Cramér–Rao bound (no λ). **Generalized profiling** and the "
           "**penalized outer** objective are both tracking estimators built on the same "
           "smoothed inner problem, and both are M-estimators whose covariance is a sandwich — "
           "profiling is *not* an NLS problem at finite λ, contrary to the note in "
           "`docs/param-covariance.md`. Both converge to NLS as λ → ∞. "
           "Move the operating point and design knobs; tick a parameter to hold it known.")

with st.sidebar:
    edit_ranges = st.checkbox("✎ edit slider ranges", value=False,
                              help="Reveal a min/max input pair above every slider so you can "
                                   "rescale the exploration range. Changing a range simply clamps "
                                   "that slider's value into the new bounds.")

    st.header("Operating point")
    m   = bounded_slider("m",  0.20, 2.00, 0.70, 0.05, edit=edit_ranges)
    n   = bounded_slider("n",  0.20, 2.50, 1.30, 0.05, edit=edit_ranges)
    L0  = bounded_slider("L0", 0.02, 0.50, 0.20, 0.02, edit=edit_ranges)
    C0  = bounded_slider("C0", 0.70, 1.50, 1.00, 0.02, edit=edit_ranges,
                         help="Non-degraded value the sample would reach with no degradation. "
                              "It is NOT directly observed — the trajectory starts at Y0 < C0.")
    Y0  = bounded_slider("Y0", 0.30, 1.40, 0.96, 0.02, edit=edit_ranges,
                         help="Measured initial value: y0 = Y0, an independent parameter held "
                              "strictly between L0 and C0. The initial-condition sensitivity now "
                              "loads onto Y0 instead of C0, decoupling the two. y = C0 is a root "
                              "of the SB rhs, so Y0 must stay below C0 (and above L0) or the "
                              "trajectory degenerates.")
    Ea_kJ = bounded_slider("Ea  [kJ/mol]", 20.0, 90.0, 50.0, 2.0, edit=edit_ranges)
    kref  = bounded_slider("k_ref", 0.50, 6.00, 2.50, 0.25, edit=edit_ranges)

    st.header("Experimental design")
    n_state = bounded_slider("state nodes / temperature", 10, 80, 40, 5, edit=edit_ranges,
                             is_int=True,
                             help="Resolution of the trajectory x and the ODE penalty — the state "
                                  "discretization, independent of how much data you have. This is "
                                  "the grid the inner problem lives on.")
    n_obs = bounded_slider("observed points / temperature", 3, 80, 40, 1, edit=edit_ranges,
                           is_int=True,
                           help="How many of the state nodes actually carry data. Clamped to the "
                                "number of state nodes. Lower this to make the data sparse without "
                                "coarsening the ODE representation: SSE = ½‖O x − Y‖², so the "
                                "data-fit Hessian block is OᵀO (1 at observed nodes, 0 elsewhere), "
                                "not the identity. Set equal to the state nodes for the "
                                "fully-observed case.")
    T_max  = bounded_slider("max temperature [°C]", 25, 60, 37, 2, edit=edit_ranges, is_int=True)

    st.header("Tracking penalty")
    log_lam = bounded_slider("log₁₀ λ", -3.0, 6.0, 0.0, 0.25, edit=edit_ranges,
                             help="Weight of the ODE penalty in the inner objective. The NLS panel "
                                  "does not depend on it. As λ → ∞ the inner problem is forced onto "
                                  "the ODE manifold and both tracking estimators converge to NLS, "
                                  "so all three panels must coincide; push the slider right to "
                                  "check.")
    lam = 10.0 ** log_lam
    st.caption(f"λ = {lam:.3g}. σ² is a common factor in all three covariances and cancels out "
               "of the correlations, so it is not a knob here. Beyond λ ≈ 10³ the three panels "
               "agree to ~1e-4. The trΣ/trΣ₀ figure on each panel is the efficiency loss "
               "against the Cramér–Rao bound — watch it blow up for profiling as λ → 0, where "
               "the smoother fits the data for any θ and identification is lost.")

    st.header("Fix (hold known)")
    fixed = {k: st.checkbox(f"fix {k}") for k in PKEYS}

free = tuple(k for k in PKEYS if not fixed[k])
temps = tuple(np.linspace(5.0, float(T_max), 6))

if not (L0 < Y0 < C0):
    st.warning(f"Need L0 < Y0 < C0 (got L0={L0:g}, Y0={Y0:g}, C0={C0:g}). The trajectory starts at "
               "y0 = Y0; at y = C0 it hits a root of the SB rhs and at y = L0 it is clipped, so "
               "outside this range the solution degenerates and the sensitivities are unreliable.")
elif len(free) < 2:
    st.warning("Leave at least two parameters free to form a correlation matrix.")
else:
    corrs, eff, free_list = compute(m, n, L0, C0, Y0, Ea_kJ * 1e3, kref,
                                    temps, n_state, n_obs, 8.0, lam, free)
    fig = make_figure(corrs, eff, free_list,
                      m=m, n=n, L0=L0, C0=C0, Y0=Y0, Ea=Ea_kJ * 1e3, kref=kref,
                      temps=temps, n_state=n_state, n_obs=n_obs, lam=lam)
    st.pyplot(fig)
