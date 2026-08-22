"""
General ODE parameter-correlation explorer — self-hosted Streamlit dashboard.

The companion of `sb_correlation.py`, which is hard-wired to the scalar Šesták–Berggren model.
This module keeps the *same* three covariance estimators but lets you type in an arbitrary
d-dimensional ODE system:

    dx/dt = f(x, θ; t),   x ∈ ℝ^d,   x(0) = y0(θ),   θ ∈ ℝ^P

A single trajectory is integrated, observed on a sparse grid of time nodes and a chosen subset of
state components, and the three estimators of the parameter covariance are compared:

  fisher()    — NLS on the exact ODE solution.  Σ₀ = σ²(SᵀS)⁻¹, S = O·∂x*/∂θ.  Cramér–Rao bound.
  profiling() — generalized profiling.  H(θ) = ½‖O x_θ − Y‖², penalty only in the inner problem.
  sandwich()  — penalized outer objective (envelope theorem).  H̃(θ) = J(x_θ, θ).

The maths is identical to `sb_correlation.py`; see docs/param-covariance.md. The covariance
formulas are written purely in the residual blocks R (∂r/∂x), G (∂r/∂θ) and the observation
operator O, so they are dimension-agnostic. The only thing that changes for a d-dim system is how
R and G are assembled:

    r_0 = x_0 − y0(θ) ∈ ℝ^d,      r_i = x_i − Φ(x_{i−1}; θ) ∈ ℝ^d
    X = [x_0; …; x_{N−1}] ∈ ℝ^{Nd}        (node-major flattening)
    R ∈ ℝ^{Nd×Nd}: block lower-bidiagonal, I_d on the diagonal, −∂Φ/∂x (d×d) on the subdiagonal
    G ∈ ℝ^{Nd×P}:  block-row 0 = −∂y0/∂θ, block-row i = −∂Φ/∂θ (d×P)
    O ∈ ℝ^{m×Nd}:  selects observed (node, component) pairs;  SSE = ½‖O X − Y‖²

Sensitivities use an additive relative finite-difference step h_j = 1e-4·max(|θ_j|, 1), applied
identically in fisher() and _blocks() (needed for the λ→∞ agreement). Reported correlations are
invariant to that diagonal rescaling, so it only removes the positivity assumption the SB module's
multiplicative log-step made.

λ → ∞: the inner problem is forced onto the ODE manifold and all three coincide (validated to
~1e-9 on a linear system — sharper than SB's ~5e-5 because a linear f makes the finite differences
essentially exact).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st


REFINE = 10          # RK4 substeps per state interval (Φ accuracy)


class ModelError(Exception):
    """User-facing error in the typed-in model (parse/eval/divergence). Surfaced via st.error."""


# ── Runge–Kutta ────────────────────────────────────────────────────────────
def rk4_solve(rhs_fn, y0, times):
    ns, nv = len(times), len(y0)
    y = np.zeros((ns, nv))
    y[0] = y0
    for i in range(ns - 1):
        dt = times[i + 1] - times[i]
        t = times[i]
        k1 = np.asarray(rhs_fn(t, y[i]))
        k2 = np.asarray(rhs_fn(t + dt / 2, y[i] + dt / 2 * k1))
        k3 = np.asarray(rhs_fn(t + dt / 2, y[i] + dt / 2 * k2))
        k4 = np.asarray(rhs_fn(t + dt, y[i] + dt * k3))
        y[i + 1] = y[i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


# ── model compilation ──────────────────────────────────────────────────────
# Whitelisted numpy functions/constants available inside the RHS and y0 expressions.
_ALLOWED = {name: getattr(np, name) for name in (
    "exp log log10 log1p sqrt cbrt sin cos tan arcsin arccos arctan arctan2 sinh cosh tanh "
    "abs sign floor ceil where clip maximum minimum fmax fmin power heaviside pi e"
).split()}


def _check_name(name, kind):
    if not name.isidentifier():
        raise ModelError(f"{kind} name {name!r} is not a valid identifier.")
    if name in _ALLOWED or name == "t":
        raise ModelError(f"{kind} name {name!r} collides with a reserved name (a math function or "
                         f"the time variable t). Rename it.")


def compile_model(state_names, param_names, rhs_exprs, y0_exprs):
    """Compile a d-dim model from named-variable expression strings.

    Returns (f_batch, y0_of):
        f_batch(X, θ, t=0.) : X (M,d) → (M,d), the RHS evaluated elementwise over the leading axis
                              (so one Φ call integrates all state intervals at once).
        y0_of(θ)            : θ → ℝ^d initial condition.

    Expressions are evaluated in a namespace holding only the whitelisted math functions, the state
    names, the parameter names and t — no builtins. Dunder (`__…__`) names are rejected as a cheap
    guard against sandbox escapes; this is a self-hosted single-user tool, not a public endpoint."""
    state_names = list(state_names)
    param_names = list(param_names)
    d, P = len(state_names), len(param_names)
    if d == 0:
        raise ModelError("Define at least one state variable.")
    if P == 0:
        raise ModelError("Define at least one parameter.")
    if len(set(state_names)) != d:
        raise ModelError("State variable names must be unique.")
    if len(set(param_names)) != P:
        raise ModelError("Parameter names must be unique.")
    for s in state_names:
        _check_name(s, "State")
    for p in param_names:
        _check_name(p, "Parameter")
    if set(state_names) & set(param_names):
        raise ModelError("A name is used for both a state variable and a parameter.")

    def _compile(exprs, tag):
        codes = []
        for nm, e in zip(state_names, exprs):
            e = (e or "").strip()
            if e == "":
                raise ModelError(f"{tag} for {nm!r} is empty.")
            if "__" in e:
                raise ModelError(f"{tag} for {nm!r} contains '__', which is not allowed.")
            try:
                codes.append(compile(e, f"<{tag}[{nm}]>", "eval"))
            except SyntaxError as ex:
                raise ModelError(f"Syntax error in {tag} for {nm!r}: {ex.msg}")
        return codes

    rhs_code = _compile(rhs_exprs, "dx/dt")
    y0_code = _compile(y0_exprs, "y0")

    def f_batch(X, theta, t=0.0):
        X = np.atleast_2d(np.asarray(X, float))
        env = dict(_ALLOWED)
        for c, s in enumerate(state_names):
            env[s] = X[:, c]
        for j, p in enumerate(param_names):
            env[p] = float(theta[j])
        env["t"] = t
        out = np.empty_like(X)
        for c, code in enumerate(rhs_code):
            try:
                out[:, c] = eval(code, {"__builtins__": {}}, env)     # noqa: S307 (sandboxed)
            except Exception as ex:
                raise ModelError(f"Error evaluating d{state_names[c]}/dt: {ex}")
        return out

    def y0_of(theta):
        env = dict(_ALLOWED)
        for j, p in enumerate(param_names):
            env[p] = float(theta[j])
        env["t"] = 0.0
        out = np.empty(d)
        for c, code in enumerate(y0_code):
            try:
                out[c] = float(eval(code, {"__builtins__": {}}, env))  # noqa: S307 (sandboxed)
            except Exception as ex:
                raise ModelError(f"Error evaluating y0[{state_names[c]}]: {ex}")
        return out

    return f_batch, y0_of


# ── grids, observation operator, trajectory, flow map ──────────────────────
def _grids(t_h, n_state):
    n_fine = (n_state - 1) * REFINE + 1
    t_fine = np.linspace(0.0, t_h, n_fine)
    return t_fine, np.arange(0, n_fine, REFINE)


def _obs_operator(n_state, n_obs, d, obs_comps):
    """O (m × n_state·d): one row per observed (node, component) pair, node-major.

    Observed nodes are n_obs values spread uniformly across the state grid; observed components are
    obs_comps. n_obs = n_state and obs_comps = all gives full observation (O has one 1 per state)."""
    n_obs = min(int(n_obs), int(n_state))
    nodes = np.unique(np.linspace(0, n_state - 1, n_obs).round().astype(int))
    obs_comps = sorted(obs_comps)
    rows = [n * d + c for n in nodes for c in obs_comps]
    O = np.zeros((len(rows), n_state * d))
    O[np.arange(len(rows)), rows] = 1.0
    return O


def _trajectory(f_batch, y0_of, theta, t_fine, idx, d):
    """Integrate the d-dim system; return the sampled nodes flattened node-major → ℝ^{Nd}."""
    rhs = lambda tt, z: f_batch(z, theta, tt)[0]
    Y = rk4_solve(rhs, y0_of(theta), t_fine)[idx]          # (n_state, d)
    return Y.reshape(-1)


def _flow_map(f_batch, Xp, theta, dt_sub):
    """Φ over one state interval, batched over the leading axis: Xp (M,d) → (M,d)."""
    y = np.array(Xp, float)
    for _ in range(REFINE):
        k1 = f_batch(y, theta)
        k2 = f_batch(y + dt_sub / 2 * k1, theta)
        k3 = f_batch(y + dt_sub / 2 * k2, theta)
        k4 = f_batch(y + dt_sub * k3, theta)
        y = y + dt_sub / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def _fd_step(theta):
    """Additive relative FD step, shared by fisher() and _blocks() so −R⁻¹G matches S at λ→∞."""
    return 1e-4 * np.maximum(np.abs(np.asarray(theta, float)), 1.0)


# ── Strategy 0: Fisher information of the exact ODE solution ────────────────
def fisher(f_batch, y0_of, theta, *, t_h, n_state, n_obs, d, obs_comps, free_idx):
    theta = np.asarray(theta, float)
    t_fine, idx = _grids(t_h, n_state)
    O = _obs_operator(n_state, n_obs, d, obs_comps)
    h = _fd_step(theta)
    S = np.zeros((O.shape[0], len(free_idx)))
    for c, j in enumerate(free_idx):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h[j]; tm[j] -= h[j]
        xp = O @ _trajectory(f_batch, y0_of, tp, t_fine, idx, d)
        xm = O @ _trajectory(f_batch, y0_of, tm, t_fine, idx, d)
        S[:, c] = (xp - xm) / (2 * h[j])
    return _safe_inv(S.T @ S)          # σ² drops out of the correlation


# ── shared inner blocks (state grid) ───────────────────────────────────────
def _blocks(f_batch, y0_of, theta, *, t_h, n_state, d, free_idx):
    """Block-bidiagonal R (Nd×Nd) and block G (Nd×P_free) of the collocation residual, per the
    module docstring. Assembled at the true θ and trajectory, where r = 0, so Gauss–Newton is
    exact. Derivatives are taken only w.r.t. the free parameters (columns of G)."""
    theta = np.asarray(theta, float)
    t_fine, idx = _grids(t_h, n_state)
    dt_sub = t_fine[1] - t_fine[0]
    h = _fd_step(theta)
    Nd, Pf = n_state * d, len(free_idx)

    # ∂y0/∂θ (d × P_free)
    dy0 = np.zeros((d, Pf))
    for c, j in enumerate(free_idx):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h[j]; tm[j] -= h[j]
        dy0[:, c] = (y0_of(tp) - y0_of(tm)) / (2 * h[j])

    X = _trajectory(f_batch, y0_of, theta, t_fine, idx, d).reshape(n_state, d)
    Xp = X[:-1]                                            # interval starts, (n_state-1, d)

    # ∂Φ/∂x — per-interval d×d. Column k = ∂Φ/∂x_in[k]; central FD per input component.
    hx = 1e-6 * (np.abs(Xp).max(axis=0) + 1.0)
    dphidx = np.zeros((n_state - 1, d, d))
    for k in range(d):
        Xp_p = Xp.copy(); Xp_m = Xp.copy()
        Xp_p[:, k] += hx[k]; Xp_m[:, k] -= hx[k]
        dphidx[:, :, k] = (_flow_map(f_batch, Xp_p, theta, dt_sub)
                           - _flow_map(f_batch, Xp_m, theta, dt_sub)) / (2 * hx[k])

    # ∂Φ/∂θ — per-interval d×P_free.
    dphidth = np.zeros((n_state - 1, d, Pf))
    for c, j in enumerate(free_idx):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h[j]; tm[j] -= h[j]
        dphidth[:, :, c] = (_flow_map(f_batch, Xp, tp, dt_sub)
                            - _flow_map(f_batch, Xp, tm, dt_sub)) / (2 * h[j])

    R = np.eye(Nd)
    G = np.zeros((Nd, Pf))
    G[0:d, :] = -dy0
    for i in range(1, n_state):
        R[i * d:(i + 1) * d, (i - 1) * d:i * d] = -dphidx[i - 1]
        G[i * d:(i + 1) * d, :] = -dphidth[i - 1]
    return R, G


# ── Strategy 2: envelope-theorem sandwich ──────────────────────────────────
def sandwich(f_batch, y0_of, theta, *, t_h, n_state, n_obs, d, obs_comps, free_idx, lam):
    R, G = _blocks(f_batch, y0_of, theta, t_h=t_h, n_state=n_state, d=d, free_idx=free_idx)
    OtO = _obs_operator(n_state, n_obs, d, obs_comps)
    OtO = OtO.T @ OtO
    Jxx = OtO + lam * (R.T @ R)                            # = OᵀO + λ ∂²PEN/∂x∂xᵀ
    U = R.T @ G                                            # J_xθ = λ U
    V = np.linalg.solve(Jxx, U)                            # J_xx⁻¹ J_xθ / λ
    B = lam * (G.T @ G) - lam ** 2 * (U.T @ V)
    M = lam ** 2 * (V.T @ (OtO @ V))                       # λ² Vᵀ OᵀO V
    Binv = _safe_inv(B)
    Sigma = Binv @ M @ Binv
    return 0.5 * (Sigma + Sigma.T)


# ── Strategy 3: generalized profiling ──────────────────────────────────────
def profiling(f_batch, y0_of, theta, *, t_h, n_state, n_obs, d, obs_comps, free_idx, lam):
    R, G = _blocks(f_batch, y0_of, theta, t_h=t_h, n_state=n_state, d=d, free_idx=free_idx)
    O = _obs_operator(n_state, n_obs, d, obs_comps)
    OtO = O.T @ O
    I_obs = np.eye(O.shape[0])
    Jxx_inv = _safe_inv(OtO + lam * (R.T @ R))
    X = -lam * (Jxx_inv @ (R.T @ G))                       # dx_θ/dθ = −J_xx⁻¹ J_xθ
    A = X.T @ O.T @ (O @ Jxx_inv @ O.T - I_obs)            # dg/dY
    Bread = X.T @ (OtO @ X)
    Meat = A @ A.T
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


# ── orchestration (cached) ─────────────────────────────────────────────────
@st.cache_data
def compute(state_names, param_names, rhs_exprs, y0_exprs, theta_vals, obs_comps,
            n_state, n_obs, t_h, lam, free):
    """Returns (corrs, eff, free, warn). Raises ModelError on a bad/divergent model."""
    f_batch, y0_of = compile_model(state_names, param_names, rhs_exprs, y0_exprs)
    theta = np.asarray(theta_vals, float)
    d = len(state_names)
    free = list(free)
    free_idx = [param_names.index(k) for k in free]
    obs_comps = list(obs_comps)

    # fail fast on a divergent trajectory instead of feeding NaN into the linear algebra
    t_fine, idx = _grids(t_h, n_state)
    base = _trajectory(f_batch, y0_of, theta, t_fine, idx, d)
    if not np.all(np.isfinite(base)):
        raise ModelError("The trajectory diverged to NaN/Inf. Check the RHS, or reduce the time "
                         "horizon / parameter magnitudes.")

    kw = dict(t_h=t_h, n_state=n_state, n_obs=n_obs, d=d, obs_comps=obs_comps, free_idx=free_idx)
    C0 = fisher(f_batch, y0_of, theta, **kw)
    C3 = profiling(f_batch, y0_of, theta, lam=lam, **kw)
    C2 = sandwich(f_batch, y0_of, theta, lam=lam, **kw)

    warn = None
    diag0 = np.diag(C0)
    if not np.all(np.isfinite(diag0)) or np.any(diag0 <= 0):
        bad = [free[i] for i in range(len(free)) if not np.isfinite(diag0[i]) or diag0[i] <= 0]
        warn = ("Some parameters look unidentifiable from the observed components (singular "
                f"information matrix): {', '.join(bad)}. Their correlations show as NaN.")
    eff = [np.trace(C) / np.trace(C0) for C in (C0, C3, C2)]
    return [_corr(C) for C in (C0, C3, C2)], eff, free, warn


# ── plotting ───────────────────────────────────────────────────────────────
def _heatmap(ax, Corr, labels, title):
    k = len(labels)
    im = ax.imshow(Corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(k)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(k)); ax.set_yticklabels(labels)
    for i in range(k):
        for j in range(k):
            v = Corr[i, j]
            txt = "nan" if not np.isfinite(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color="white" if np.isfinite(v) and abs(v) > 0.5 else "black")
    ax.set_title(title, fontsize=8.5)
    return im


def _strongest(Corr, free):
    od = Corr - np.eye(len(free))
    od = np.where(np.isfinite(od), od, 0.0)
    if not np.any(od):
        return "—"
    a, b = np.unravel_index(np.argmax(np.abs(od)), od.shape)
    return f"ρ({free[a]},{free[b]}) = {od[a, b]:+.3f}"


def make_figure(corrs, eff, free, *, d, obs_comps_names, n_state, n_obs, lam, fixed):
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
        im = _heatmap(ax, Corr, free,
                      f"{title}\nstrongest {_strongest(Corr, free)}\n"
                      f"tr$\\Sigma$/tr$\\Sigma_0$ = {e:.2f}×")
    fig.colorbar(im, ax=list(axes), shrink=0.85)
    fig.suptitle(f"d={d}-dim system   |   observed: {', '.join(obs_comps_names)}   "
                 f"|   {n_obs} obs nodes on {n_state}-node grid   |   "
                 f"estimated: {len(free)} params (fixed: {'none' if not fixed else ', '.join(fixed)})",
                 fontsize=10)
    return fig


# ── Streamlit UI ───────────────────────────────────────────────────────────
def bounded_slider(label, lo, hi, default, step, *, edit=False, help=None, is_int=False):
    """Slider whose min/max the user can override when `edit` is on; value is always clamped in."""
    cast = int if is_int else float
    fmt = "%d" if is_int else None
    if edit:
        c1, c2 = st.columns(2)
        lo = c1.number_input(f"{label} min", value=cast(lo), step=cast(step), format=fmt,
                             key=f"{label}__lo")
        hi = c2.number_input(f"{label} max", value=cast(hi), step=cast(step), format=fmt,
                             key=f"{label}__hi")
    lo, hi, step = cast(lo), cast(hi), cast(step)
    if hi <= lo:
        hi = lo + step
    default = cast(min(max(default, lo), hi))
    return st.slider(label, lo, hi, default, step, format=fmt, help=help)


# Default model: consecutive reaction A → B → (C), a cleanly correlated (k1, k2) pair.
DEFAULT_STATES = "A, B"
DEFAULT_RHS = {"A": "-k1*A", "B": "k1*A - k2*B"}
DEFAULT_Y0 = {"A": "1.0", "B": "0.0"}
DEFAULT_PARAMS = pd.DataFrame({"name": ["k1", "k2"], "value": [1.2, 0.6]})


st.set_page_config(page_title="ODE correlation explorer", layout="wide")
st.title("Parameter-correlation explorer — general ODE")
st.caption("Type in any d-dimensional ODE system and compare three estimators of the parameter "
           "covariance: **NLS** on the exact solution (Cramér–Rao bound, no λ), **generalized "
           "profiling**, and the **penalized-outer** sandwich. Both tracking estimators are "
           "M-estimators whose covariance is a sandwich, and both converge to NLS as λ → ∞ "
           "(see `docs/param-covariance.md`). The Šesták–Berggren special case lives in "
           "`sb_correlation.py`.")

with st.sidebar:
    edit_ranges = st.checkbox("✎ edit slider ranges", value=False,
                              help="Reveal min/max inputs above every slider to rescale its range.")

    st.header("Model definition")
    state_str = st.text_input("state variable names (comma-separated)", DEFAULT_STATES,
                              help="One name per state component; the system dimension d is the "
                                   "number of names. Names must be valid identifiers and cannot "
                                   "clash with a math function or t.")
    state_names = [s.strip() for s in state_str.split(",") if s.strip()]

    st.markdown("**Parameters** — edit values; add/remove rows.")
    param_df = st.data_editor(DEFAULT_PARAMS, num_rows="dynamic", hide_index=True,
                              column_config={"name": st.column_config.TextColumn("name"),
                                             "value": st.column_config.NumberColumn("value",
                                                                                    format="%g")},
                              key="param_table")

    st.markdown("**Right-hand side**  dx/dt = f(x, θ; t)")
    rhs_exprs = [st.text_input(f"d{s}/dt =", DEFAULT_RHS.get(s, "0"), key=f"rhs_{s}")
                 for s in state_names]

    st.markdown("**Initial condition**  x(0) = y0(θ)")
    y0_exprs = [st.text_input(f"y0[{s}] =", DEFAULT_Y0.get(s, "0"), key=f"y0_{s}")
                for s in state_names]

    obs_names = st.multiselect("observed components", state_names, default=state_names,
                               help="Which state components carry data. Deselect to explore "
                                    "partial observation; parameters unidentifiable from the "
                                    "observed set show NaN correlations.")

    st.header("Experimental design")
    t_h = bounded_slider("time horizon", 1.0, 40.0, 8.0, 1.0, edit=edit_ranges)
    n_state = bounded_slider("state nodes", 10, 80, 40, 5, edit=edit_ranges, is_int=True,
                             help="Resolution of the trajectory and ODE penalty — the state "
                                  "discretization, independent of how much data you have.")
    n_obs = bounded_slider("observed time nodes", 3, 80, 40, 1, edit=edit_ranges, is_int=True,
                           help="How many time nodes carry data (clamped to the state nodes). "
                                "Combined with the observed components, this is the sparsity of "
                                "the data: SSE = ½‖O X − Y‖² with O selecting (node, component).")

    st.header("Tracking penalty")
    log_lam = bounded_slider("log₁₀ λ", -3.0, 6.0, 0.0, 0.25, edit=edit_ranges,
                             help="Weight of the ODE penalty in the inner objective. NLS does not "
                                  "depend on it; as λ → ∞ all three panels coincide.")
    lam = 10.0 ** log_lam
    st.caption(f"λ = {lam:.3g}. σ² cancels out of the correlations. trΣ/trΣ₀ on each panel is the "
               "efficiency loss vs the Cramér–Rao bound; watch profiling blow up as λ → 0.")

    # parameter names come from the (possibly edited) table
    param_names = [str(x).strip() for x in param_df["name"].tolist() if str(x).strip()]
    st.header("Fix (hold known)")
    fixed_flags = {k: st.checkbox(f"fix {k}", key=f"fix_{k}") for k in param_names}


# ── assemble + render ──────────────────────────────────────────────────────
def _run():
    if not state_names:
        st.warning("Define at least one state variable."); return
    if not param_names:
        st.warning("Define at least one parameter in the table."); return
    try:
        theta_vals = [float(v) for v in param_df["value"].tolist()][:len(param_names)]
    except (TypeError, ValueError):
        st.error("Every parameter needs a numeric value."); return
    if len(theta_vals) != len(param_names):
        st.warning("Give each parameter a name and a value."); return

    free = tuple(k for k in param_names if not fixed_flags.get(k, False))
    if len(free) < 2:
        st.warning("Leave at least two parameters free to form a correlation matrix."); return

    obs_comps = tuple(state_names.index(s) for s in obs_names)
    if not obs_comps:
        st.warning("Select at least one observed component."); return

    try:
        corrs, eff, free_list, warn = compute(
            tuple(state_names), tuple(param_names), tuple(rhs_exprs), tuple(y0_exprs),
            tuple(theta_vals), obs_comps, n_state, n_obs, float(t_h), lam, free)
    except ModelError as ex:
        st.error(f"⚠️ {ex}"); return

    if warn:
        st.warning(warn)
    fixed = [k for k in param_names if fixed_flags.get(k, False)]
    fig = make_figure(corrs, eff, free_list, d=len(state_names), obs_comps_names=list(obs_names),
                      n_state=n_state, n_obs=n_obs, lam=lam, fixed=fixed)
    st.pyplot(fig)


_run()
