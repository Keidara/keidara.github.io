"""Oscillateur harmonique amorti, libre ou forcé : solution exacte, portrait de phase, résonance.

Équation : x'' + 2ζω₀ x' + ω₀² x = ω₀² F cos(ωt). Le temps est mesuré en périodes propres T₀ = 2π/ω₀
(donc ω₀ = 2π), la vitesse en unités de ω₀, et F est le déplacement statique F₀/k.
"""
import numpy as np
from matplotlib.patches import Rectangle
from kit import *

PARAMS = [
    Select("mode", "Mode", ["Libre", "Forcé"]),
    Slider("zeta", "Taux d’amortissement ζ", 0.01, 2.0, 0.1, step=0.01),
    Slider("x0", "Position initiale x₀", -1.0, 1.0, 1.0, step=0.05),
    Slider("v0", "Vitesse initiale v₀ / ω₀", -1.0, 1.0, 0.0, step=0.05),
    Slider("omega", "Fréquence de la force ω / ω₀", 0.1, 3.0, 0.9, step=0.01, actif=("mode", ["Forcé"])),
    Slider("F", "Amplitude de la force F₀ / k", 0.0, 2.0, 0.5, step=0.05, actif=("mode", ["Forcé"])),
    Slider("t", "Temps t / T₀", 0.0, 20.0, 0.0, step=0.05, anim=20),
]

W0 = 2 * np.pi          # ω₀ en rad par période T₀
TMAX = 20.0             # durée affichée, en périodes


def reponse(r, zeta):
    """Amplitude relative X / F et déphasage φ du régime permanent, pour r = ω/ω₀."""
    r = np.asarray(r, float)
    den = np.maximum(np.sqrt((1 - r ** 2) ** 2 + (2 * zeta * r) ** 2), 1e-9)
    return 1 / den, np.arctan2(2 * zeta * r, 1 - r ** 2)


def solution(t, mode, zeta, x0, v0, omega, F):
    """Position x(t) et vitesse v(t)/ω₀, solution exacte avec les conditions initiales données."""
    t = np.asarray(t, float)
    w = omega * W0
    xp = vp = 0.0
    xp0 = vp0 = 0.0
    if mode == "Forcé":
        A, phi = reponse(omega, zeta)
        X = F * float(A)
        phi = float(phi)
        xp, vp = X * np.cos(w * t - phi), -X * w * np.sin(w * t - phi)       # régime permanent
        xp0, vp0 = X * np.cos(phi), X * w * np.sin(phi)
    a, b = x0 - xp0, v0 * W0 - vp0                                           # conditions initiales du régime transitoire
    if abs(zeta - 1) < 1e-3:                                                 # amortissement critique
        c = b + W0 * a
        e = np.exp(-W0 * t)
        xh, vh = (a + c * t) * e, (c - W0 * (a + c * t)) * e
    else:
        rac = np.sqrt(complex(zeta ** 2 - 1))
        s1, s2 = W0 * (-zeta + rac), W0 * (-zeta - rac)
        c1 = (b - s2 * a) / (s1 - s2)
        c2 = a - c1
        e1, e2 = np.exp(s1 * t), np.exp(s2 * t)
        xh, vh = (c1 * e1 + c2 * e2).real, (c1 * s1 * e1 + c2 * s2 * e2).real
    return xh + xp, (vh + vp) / W0


def regime(zeta):
    if abs(zeta - 1) < 1e-3:
        return "amortissement critique (ζ = 1) : retour le plus rapide sans oscillation"
    if zeta < 1:
        return f"sous-amorti (ζ = {zeta:.2f}) : pseudo-période {1 / np.sqrt(1 - zeta ** 2):.3f} T₀"
    return f"sur-amorti (ζ = {zeta:.2f}) : retour lent sans oscillation"


def ressort(x_debut, x_fin, n=9, h=0.28):
    """Points d'un ressort en zigzag entre deux abscisses."""
    xs = np.linspace(x_debut, x_fin, 2 * n + 3)
    ys = np.zeros_like(xs)
    ys[1:-1:2] = h
    ys[2:-1:2] = -h
    return xs, ys


def fig_mouvement(fig, mode, zeta, x0, v0, omega, F, t):
    tt = np.linspace(0, TMAX, 1500)
    x, _ = solution(tt, mode, zeta, x0, v0, omega, F)
    xm = max(1.15 * float(np.abs(x).max()), 0.3)                  # échelle fixe pendant l'animation
    xt = float(solution(t, mode, zeta, x0, v0, omega, F)[0])
    ax_s, ax = fig.subplots(2, 1, gridspec_kw={"height_ratios": [1, 3]})

    mur, h = -2 * xm, 0.13 * xm
    ax_s.set_xlim(-2.2 * xm, 1.4 * xm)
    ax_s.set_ylim(-1, 1)
    ax_s.axis("off")
    ax_s.plot([mur, mur], [-0.7, 0.7], color="0.4", lw=4)
    ax_s.axvline(0, ls=":", color="0.5", lw=1)
    ax_s.plot(*ressort(mur, xt - h), color="0.5", lw=1.8)
    ax_s.add_patch(Rectangle((xt - h, -0.4), 2 * h, 0.8, facecolor="C0", edgecolor="none"))

    ax.plot(tt, x, color="C0", lw=2, label="position x(t)")
    if mode == "Forcé":
        A, phi = reponse(omega, zeta)
        ax.plot(tt, F * float(A) * np.cos(omega * W0 * tt - float(phi)), "--", color="C1", lw=1.3, label="régime permanent")
    ax.axvline(t, color="0.5", lw=1)
    ax.plot([t], [xt], "o", color="C3", ms=9)
    ax.set_xlim(0, TMAX)
    ax.set_ylim(-xm, xm)
    ax.set_xlabel("Temps t / T₀")
    ax.set_ylabel("Position x")
    ax.set_title(regime(zeta))
    ax.legend(loc="lower right", fontsize=9)


def fig_phase(fig, mode, zeta, x0, v0, omega, F):
    ax = fig.subplots()
    tt = np.linspace(0, TMAX, 4000)
    x, v = solution(tt, mode, zeta, x0, v0, omega, F)
    ax.plot(x, v, color="C0", lw=1.5)
    ax.plot([x[0]], [v[0]], "o", color="C2", ms=8, label="départ")
    ax.plot([x[-1]], [v[-1]], "s", color="C3", ms=7, label=f"t = {TMAX:g} T₀")
    ax.set_aspect("equal", "datalim")
    ax.set_xlabel("Position x")
    ax.set_ylabel("Vitesse v / ω₀")
    ax.set_title("Portrait de phase")
    ax.legend(loc="upper right", fontsize=9)


def fig_resonance(fig, mode, zeta, omega):
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    r = np.linspace(0.01, 3, 600)
    for z in (0.05, 0.1, 0.25, 0.5, 1.0):
        ax1.plot(r, reponse(r, z)[0], color="0.7", lw=1)
        ax2.plot(r, reponse(r, z)[1], color="0.7", lw=1)
    ax1.plot([], [], color="0.7", label="autres ζ (0,05 ; 0,1 ; 0,25 ; 0,5 ; 1)")
    a, p = reponse(r, zeta)
    ax1.plot(r, a, color="C0", lw=2.2, label=f"ζ = {zeta:.2f}")
    ax2.plot(r, p, color="C0", lw=2.2)
    if mode == "Forcé":
        ra, pa = reponse(omega, zeta)
        ax1.plot([omega], [ra], "o", color="C3", ms=9, label=f"ω / ω₀ = {omega:.2f}")
        ax2.plot([omega], [pa], "o", color="C3", ms=9)
    for ax in (ax1, ax2):
        ax.axvline(1, ls=":", color="0.5")
    ax1.set_yscale("log")
    ax1.set_ylim(0.03, 100)
    ax1.set_ylabel("Amplitude X / (F₀/k)")
    if zeta < 1 / np.sqrt(2):
        rres = np.sqrt(1 - 2 * zeta ** 2)
        ax1.set_title(f"Résonance : maximum à ω/ω₀ = {rres:.3f}, amplitude {1 / (2 * zeta * np.sqrt(1 - zeta ** 2)):.3g} fois F₀/k")
    else:
        ax1.set_title("Pas de résonance (ζ ≥ 0,707) : l’amplitude ne fait que décroître")
    ax1.legend(loc="upper right", fontsize=8)
    ax2.set_ylim(-0.1, np.pi + 0.1)
    ax2.set_yticks([0, np.pi / 2, np.pi])
    ax2.set_yticklabels(["0", "π/2", "π"])
    ax2.set_ylabel("Retard de phase φ")
    ax2.set_xlabel("Fréquence de la force ω / ω₀")
    ax2.set_xlim(0, 3)
