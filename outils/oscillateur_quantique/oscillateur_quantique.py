"""Oscillateur harmonique quantique : états propres et état cohérent (unités : ħ = m = ω = 1, x en unités de √(ħ/mω))."""
import numpy as np
from math import lgamma
from kit import *

PARAMS = [
    Select("mode", "État", ["État propre", "État cohérent"]),
    Slider("n", "Niveau n", 0, 12, 2, actif=("mode", ["État propre"])),
    Slider("alpha", "Amplitude α", 0.5, 4.0, 2.0, step=0.1, actif=("mode", ["État cohérent"])),
    Slider("t", "Temps t / T", 0.0, 1.0, 0.0, step=0.005, anim=8, actif=("mode", ["État cohérent"])),
]

NMAX = 70


def etats_propres(N, x):
    """ψ_0 ... ψ_N sur la grille x, par récurrence (stable)."""
    p = np.zeros((N + 1, len(x)))
    p[0] = np.pi ** -0.25 * np.exp(-x ** 2 / 2)
    if N >= 1:
        p[1] = np.sqrt(2) * x * p[0]
    for k in range(1, N):
        p[k + 1] = np.sqrt(2 / (k + 1)) * x * p[k] - np.sqrt(k / (k + 1)) * p[k - 1]
    return p


def fig_etat(fig, mode, n, alpha, t):
    if mode == "État propre":
        demi = max(6.0, np.sqrt(2 * n + 1) + 3)
    else:
        demi = np.sqrt(2) * alpha + 4
    x = np.linspace(-demi, demi, 900)
    ax_v, ax_p = fig.subplots(2, 1, sharex=True)

    if mode == "État propre":
        dens = etats_propres(n, x)[n] ** 2
        E = n + 0.5
        top = max(E * 1.35, 3.0)
        for k in range(0, max(n + 3, 6)):
            ax_v.axhline(k + 0.5, color="0.7", lw=0.8)
        ax_v.axhline(E, color="C1", lw=2.5, label=f"E = {E:g} ħω")
        xt = np.sqrt(2 * E)
        ax_v.plot([-xt, xt], [E, E], "o", color="C1", ms=6)
        cl = np.where(2 * E - x ** 2 > 0, 1 / (np.pi * np.sqrt(np.maximum(2 * E - x ** 2, 1e-12))), np.nan)
        ax_p.plot(x, cl, "--", color="C1", lw=1.5, label="particule classique de même énergie")
        titre = f"État propre n = {n} : E = {E:g} ħω"
        ymax = 1.4 * dens.max()
    else:
        N = NMAX
        n_ = np.arange(N + 1)
        lg = np.array([lgamma(k + 1) for k in n_])
        c = np.exp(-alpha ** 2 / 2 + n_ * np.log(alpha) - 0.5 * lg)
        psi = (c * np.exp(-1j * (n_ + 0.5) * 2 * np.pi * t)) @ etats_propres(N, x)
        dens = np.abs(psi) ** 2
        xc = np.sqrt(2) * alpha * np.cos(2 * np.pi * t)             # position de la particule classique
        top = max(alpha ** 2 * 1.5 + 1, 3.0)
        ax_v.axhline(alpha ** 2 + 0.5, color="C1", lw=2, label=f"⟨E⟩ = {alpha ** 2 + 0.5:.2f} ħω")
        ax_v.plot([xc], [xc ** 2 / 2], "o", color="C3", ms=10, label="particule classique")
        ax_p.axvline(xc, ls="--", color="C3", lw=1.5, label="position classique")
        titre = f"État cohérent α = {alpha:.1f} : le paquet oscille sans se déformer"
        ymax = 0.75                                                   # pic de |ψ|² = 1/√π ≈ 0,56, constant

    ax_v.plot(x, x ** 2 / 2, color="C0", lw=2)
    ax_v.set_ylim(0, top)
    ax_v.set_ylabel("Énergie (ħω)")
    ax_v.set_title(titre)
    ax_v.legend(loc="upper center", fontsize=9)
    ax_p.fill_between(x, dens, color="C2", alpha=0.35)
    ax_p.plot(x, dens, color="C2", lw=2, label="densité de probabilité |ψ|²")
    ax_p.set_ylim(0, ymax)
    ax_p.set_xlim(-demi, demi)
    ax_p.set_xlabel("Position x / x₀")
    ax_p.set_ylabel("|ψ|²")
    ax_p.legend(loc="upper right", fontsize=9)
