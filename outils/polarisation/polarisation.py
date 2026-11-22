"""Polarisation par réflexion : formules de Fresnel, angle de Brewster, lumière réfléchie vue à travers un polariseur."""
import numpy as np
from kit import *

PARAMS = [
    Slider("n1", r"Indice du milieu 1 (haut), \(n_1\)", 1.0, 2.5, 1.0, step=0.01),
    Slider("n2", r"Indice du milieu 2 (bas), \(n_2\)", 1.0, 2.5, 1.5, step=0.01),
    Slider("theta", r"Angle d’incidence \(\theta_1\)", 0, 89, 56, unit="°"),
    Slider("alpha", r"Orientation du polariseur \(\alpha\)", 0, 180, 90, unit="°"),
]


def reflexion(n1, n2, th):
    """Facteurs de réflexion en intensité (polarisation s, polarisation p, moyenne), pour des angles th en radians."""
    th = np.asarray(th, float)
    s2 = n1 * np.sin(th) / n2
    tir = s2 > 1
    c2 = np.sqrt(1 - np.minimum(s2, 1) ** 2)
    c1 = np.cos(th)
    rs = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)
    rp = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)
    Rs, Rp = np.where(tir, 1.0, rs ** 2), np.where(tir, 1.0, rp ** 2)
    return Rs, Rp, (Rs + Rp) / 2


def fig_fresnel(fig, n1, n2, theta):
    ax = fig.subplots()
    a = np.linspace(0, 89.9, 600)
    Rs, Rp, Rm = reflexion(n1, n2, np.radians(a))
    ax.plot(a, 100 * Rs, color="C0", lw=2, label="polarisation s (perpendiculaire au plan d’incidence)")
    ax.plot(a, 100 * Rp, color="C1", lw=2, label="polarisation p (dans le plan d’incidence)")
    ax.plot(a, 100 * Rm, "--", color="0.5", lw=1.5, label="lumière non polarisée")
    r0 = float(reflexion(n1, n2, np.radians([float(theta)]))[2][0])
    ax.plot([theta], [100 * r0], "o", color="C3", ms=10)
    if abs(n1 - n2) > 1e-9:
        ab = np.degrees(np.arctan(n2 / n1))
        ax.axvline(ab, ls=":", color="C1")
        ax.text(ab, 60, f" angle de Brewster {ab:.1f}°", color="C1", rotation=90, va="center", ha="right", fontsize=9)
    if n1 > n2:
        thc = np.degrees(np.arcsin(n2 / n1))
        ax.axvline(thc, ls="--", color="C2")
        ax.text(thc, 60, f" angle limite {thc:.1f}°", color="C2", rotation=90, va="center", ha="right", fontsize=9)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 100)
    ax.set_xlabel(r"Angle d’incidence $\theta_1$ (°)")
    ax.set_ylabel("Lumière réfléchie (%)")
    ax.set_title(rf"$\theta_1$ = {theta}° : {100 * r0:.1f} % réfléchi, {100 * (1 - r0):.1f} % transmis (lumière non polarisée)")
    ax.legend(loc="center left", fontsize=9)


def fig_polariseur(fig, n1, n2, theta, alpha):
    """Lumière réfléchie (lumière incidente non polarisée) à travers un polariseur d'orientation α.
    α = 0 : le polariseur laisse passer la polarisation s ; α = 90° : la polarisation p."""
    ax = fig.subplots()
    Rs, Rp, Rm = reflexion(n1, n2, np.radians([float(theta)]))
    Rs, Rp, Rm = float(Rs[0]), float(Rp[0]), float(Rm[0])
    a = np.linspace(0, 180, 361)
    ca2, sa2 = np.cos(np.radians(a)) ** 2, np.sin(np.radians(a)) ** 2
    I = 50 * (Rs * ca2 + Rp * sa2)                      # % de l'intensité incidente (moitié en s, moitié en p)
    Ia = 50 * (Rs * np.cos(np.radians(alpha)) ** 2 + Rp * np.sin(np.radians(alpha)) ** 2)
    ax.plot(a, I, color="C0", lw=2, label="lumière réfléchie après le polariseur")
    ax.axhline(100 * Rm, ls="--", color="0.5", label="sans polariseur")
    ax.plot([alpha], [Ia], "o", color="C3", ms=10)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlim(0, 180)
    ax.set_ylim(0, max(float(I.max()), 100 * Rm, 1e-3) * 1.25)
    ax.set_xlabel(r"Orientation du polariseur $\alpha$ (°)")
    ax.set_ylabel("Part de la lumière incidente (%)")
    ax.set_title(rf"$\alpha$ = {alpha}° : {Ia:.2f} % de la lumière incidente passe (sans polariseur : {100 * Rm:.2f} %)")
    ax.legend(loc="upper right", fontsize=9)
