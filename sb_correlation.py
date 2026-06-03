"""
Šesták–Berggren parameter-correlation explorer — self-hosted Streamlit dashboard.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

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

R_GAS, EPS = 8.314, 1e-6
PKEYS  = ["A", "Ea", "m", "n", "L0", "C0"]
PLABEL = {"A": "A", "Ea": r"$E_a$", "m": "m", "n": "n", "L0": r"$L_0$", "C0": r"$C_0$"}


# ---------------------------------------------------------------------------
# Model + Fisher information  (verbatim from the notebook)
# ---------------------------------------------------------------------------
def _outputs(theta, T_K, y0, t, idx):
    A, Ea, m, n, L0, C0 = theta
    out = []
    for Tk in T_K:
        k = A * np.exp(-Ea / (R_GAS * Tk))
        def rhs(tt, z, k=k):
            y = min(max(z[0], L0 + EPS), C0 - EPS)
            return [-k / (C0 - L0) ** (m + n - 1) * (C0 - y) ** m * (y - L0) ** n]
        out.append(rk4_solve(rhs, [y0], t)[:, 0][idx])
    return np.concatenate(out)


def fisher(theta, temps_C, *, free=None, n_data=40, noise=0.01, t_h=8.0, alpha0=0.05):
    free = list(PKEYS if free is None else free)
    theta = np.asarray(theta, float)
    A, Ea, m, n, L0, C0 = theta
    T_K = np.asarray(temps_C, float) + 273.15
    y0  = C0 - alpha0 * (C0 - L0)
    t   = np.linspace(0.0, t_h, 400).astype(np.float32)
    idx = np.linspace(0, len(t) - 1, n_data).astype(int)

    eps = 1e-4
    S = np.zeros((n_data * len(T_K), 6))
    for j in range(6):
        tp, tm = theta.copy(), theta.copy()
        tp[j] *= (1 + eps); tm[j] *= (1 - eps)
        S[:, j] = (_outputs(tp, T_K, y0, t, idx) - _outputs(tm, T_K, y0, t, idx)) / (2 * eps)

    cols = [PKEYS.index(k) for k in free]
    Sf = S[:, cols]
    FIM = Sf.T @ Sf / noise ** 2
    try:
        Cov = np.linalg.inv(FIM)
    except np.linalg.LinAlgError:
        Cov = np.linalg.pinv(FIM)
    se = np.sqrt(np.clip(np.diag(Cov), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        Corr = Cov / np.outer(se, se)
    return Corr, se, np.linalg.cond(FIM), free


def theta_from_knobs(m, n, L0, C0, Ea, kref, temps_C):
    T_ref = float((np.asarray(temps_C, float) + 273.15).mean())
    A = kref * np.exp(Ea / (R_GAS * T_ref))
    return np.array([A, Ea, m, n, L0, C0])


# Cache the (somewhat) expensive Fisher computation across reruns. All args are hashable.
@st.cache_data
def compute(m, n, L0, C0, Ea, kref, temps, n_data, noise, t_h, free):
    theta = theta_from_knobs(m, n, L0, C0, Ea, kref, np.asarray(temps))
    return fisher(theta, np.asarray(temps), free=list(free),
                  n_data=n_data, noise=noise, t_h=t_h)


def make_figure(Corr, se, cond, free, *, m, n, L0, C0, Ea, kref, temps, n_data, noise):
    labels = [PLABEL[k] for k in free]
    k = len(free)

    fig, (axc, axs) = plt.subplots(1, 2, figsize=(11, 4.3),
                                   gridspec_kw=dict(width_ratios=[1.15, 1]))
    im = axc.imshow(Corr, vmin=-1, vmax=1, cmap="RdBu_r")
    axc.set_xticks(range(k)); axc.set_xticklabels(labels)
    axc.set_yticks(range(k)); axc.set_yticklabels(labels)
    for i in range(k):
        for j in range(k):
            axc.text(j, i, f"{Corr[i, j]:.2f}", ha="center", va="center",
                     fontsize=8, color="white" if abs(Corr[i, j]) > 0.5 else "black")
    axc.set_title("correlation matrix")
    fig.colorbar(im, ax=axc, shrink=0.85)

    y = np.arange(k)
    axs.barh(y, 100 * se, color="tab:purple", alpha=0.85)
    axs.set_yticks(y); axs.set_yticklabels(labels); axs.invert_yaxis()
    axs.set_xscale("log"); axs.set_xlabel("relative standard error  [%]")
    for yi, s in zip(y, se):
        axs.text(100 * s, yi, f"  {100*s:.2f}%", va="center", fontsize=8)
    axs.set_title("standard errors")
    axs.grid(True, axis="x", which="both", alpha=0.3)

    od = Corr - np.eye(k)
    a, b = np.unravel_index(np.argmax(np.abs(od)), od.shape)
    fixed = [x for x in PKEYS if x not in free]
    fixed_str = "none" if not fixed else ", ".join(fixed)
    fig.suptitle(f"m={m:g}  n={n:g}  L0={L0:g}  C0={C0:g}  Ea={Ea/1e3:g} kJ/mol  k_ref={kref:g}   "
                 f"|   {len(temps)} temps, N={n_data}, σ={noise:g}\n"
                 f"estimated: {k} params   (fixed: {fixed_str})    "
                 f"strongest ρ({free[a]},{free[b]}) = {od[a, b]:+.3f}    cond(FIM) = {cond:.1e}",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SB correlation explorer", layout="wide")
st.title("Parameter-correlation explorer — Šesták–Berggren")
st.caption("Local identifiability via the Fisher information. Move the operating point and "
           "design knobs; tick a parameter to hold it known (excluded from the estimation).")

with st.sidebar:
    st.header("Operating point")
    m   = st.slider("m",  0.20, 2.00, 0.70, 0.05)
    n   = st.slider("n",  0.20, 2.50, 1.30, 0.05)
    L0  = st.slider("L0", 0.02, 0.50, 0.20, 0.02)
    C0  = st.slider("C0", 0.70, 1.50, 1.00, 0.02)
    Ea_kJ = st.slider("Ea  [kJ/mol]", 20.0, 90.0, 50.0, 2.0)
    kref  = st.slider("k_ref", 0.50, 6.00, 2.50, 0.25)

    st.header("Experimental design")
    noise  = st.slider("noise σ", 0.002, 0.030, 0.010, 0.002, format="%.3f")
    n_data = st.slider("N data / temperature", 10, 80, 40, 5)
    T_max  = st.slider("max temperature [°C]", 25, 60, 37, 2)

    st.header("Fix (hold known)")
    fixed = {
        "A":  st.checkbox("fix A"),
        "Ea": st.checkbox("fix Ea"),
        "m":  st.checkbox("fix m"),
        "n":  st.checkbox("fix n"),
        "L0": st.checkbox("fix L0"),
        "C0": st.checkbox("fix C0"),
    }

free = tuple(k for k in PKEYS if not fixed[k])
temps = tuple(np.linspace(5.0, float(T_max), 6))

if len(free) < 2:
    st.warning("Leave at least two parameters free to form a correlation matrix.")
else:
    Corr, se, cond, free_list = compute(m, n, L0, C0, Ea_kJ * 1e3, kref,
                                        temps, n_data, noise, 8.0, free)
    fig = make_figure(Corr, se, cond, free_list,
                      m=m, n=n, L0=L0, C0=C0, Ea=Ea_kJ * 1e3, kref=kref,
                      temps=temps, n_data=n_data, noise=noise)
    st.pyplot(fig)
