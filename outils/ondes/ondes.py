"""Ondes : progressive, stationnaire, réflexion sur un bord, paquet d'ondes."""
import numpy as np
from kit import *

PROG, STAT, REFL, PAQ = "Progressive", "Stationnaire", "Réflexion", "Paquet d’ondes"

PARAMS = [
    Select("mode", "Mode", [PROG, STAT, REFL, PAQ]),
    Slider("lam", r"Longueur d’onde \(\lambda\)", 0.5, 3.0, 1.0, step=0.1),
    Slider("r", r"Coefficient de réflexion \(r\) (amplitude réfléchie / incidente)", 0.0, 1.0, 0.7, step=0.05,
           actif=("mode", [REFL]), cache=True),
    Check("libre", "Bord libre (sinon bord fixe)", False, actif=("mode", [REFL]), cache=True),
    Slider("N", r"Nombre d’ondes sommées \(N\)", 1, 60, 5, actif=("mode", [PAQ]), cache=True),
    Slider("larg", r"Largeur du spectre \(\sigma / k_0\)", 5, 30, 12, unit="%", actif=("mode", [PAQ]), cache=True),
    Slider("t", r"Temps \(t / T\)", 0.0, 2.0, 0.0, step=0.02, anim=8),
    Check("compo", "Montrer les deux ondes qui s’additionnent", False, actif=("mode", [STAT, REFL]), cache=True),
]


def _axes(ax, ymax):
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel("Position x")
    ax.set_ylabel("Amplitude")


def _progressive(ax, lam, t):
    x = np.linspace(0, 4, 800)
    k, w = 2 * np.pi / lam, 2 * np.pi                    # période T = 1
    ax.plot(x, np.sin(k * x), ":", color="C2", label="à t = 0")
    ax.plot(x, np.sin(k * x - w * t), color="C0", lw=2, label="onde progressive")
    ax.set_title(f"t = {t:.2f} T : la forme avance d’une longueur d’onde par période")
    ax.legend(loc="lower right", fontsize=8)
    _axes(ax, 1.15)


def _stationnaire(ax, lam, t, compo):
    x = np.linspace(0, 4, 800)
    k, w = 2 * np.pi / lam, 2 * np.pi
    aller = 0.5 * np.sin(k * x - w * t)
    retour = 0.5 * np.sin(k * x + w * t)
    if compo:
        ax.plot(x, aller, "--", color="C2", lw=1, label="onde vers la droite")
        ax.plot(x, retour, "--", color="C3", lw=1, label="onde vers la gauche")
    ax.plot(x, np.abs(np.sin(k * x)), ":", color="C1", lw=1)
    ax.plot(x, -np.abs(np.sin(k * x)), ":", color="C1", lw=1, label="enveloppe")
    ax.plot(x, aller + retour, color="C0", lw=2, label="somme (stationnaire)")
    noeuds = np.arange(0, 4 + 1e-9, lam / 2)
    ax.plot(noeuds, np.zeros_like(noeuds), "o", color="C1", ms=6, label="nœuds")
    ax.set_title(f"t = {t:.2f} T : la forme oscille sur place, entre les nœuds")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    _axes(ax, 1.15)


def _reflexion(ax, lam, t, compo, r, libre):
    """Onde incidente vers la droite, bord en x = 0. Bord fixe : onde réfléchie inversée (nœud au bord) ;
    bord libre : onde réfléchie non inversée (ventre au bord)."""
    x = np.linspace(-4, 0, 1200)
    k, w = 2 * np.pi / lam, 2 * np.pi
    s = -1.0 if libre else 1.0
    inc = np.sin(k * x - w * t)
    ref = s * r * np.sin(k * x + w * t)
    env = np.sqrt(((1 + s * r) * np.sin(k * x)) ** 2 + ((1 - s * r) * np.cos(k * x)) ** 2)
    if compo:
        ax.plot(x, inc, "--", color="C2", lw=1, label="onde incidente")
        if r > 0:
            ax.plot(x, ref, "--", color="C3", lw=1, label="onde réfléchie")
    ax.plot(x, env, ":", color="C1", lw=1, label="enveloppe")
    ax.plot(x, -env, ":", color="C1", lw=1)
    ax.plot(x, inc + ref, color="C0", lw=2, label="somme")
    if r > 0.05:                                          # emplacements où l'amplitude est minimale
        xm = -np.arange(0, int(4 / (lam / 2)) + 2) * lam / 2 - (lam / 4 if libre else 0.0)
        xm = xm[xm >= -4]
        ax.plot(xm, np.zeros_like(xm), "o", color="C1", ms=6, label="nœuds" if r > 0.995 else "minima d’amplitude")
    ax.axvspan(0, 0.35, color="0.5", alpha=0.35, hatch="//")
    ax.axvline(0, color="0.4", lw=4)
    ax.set_xlim(-4, 0.35)
    if r < 0.005:
        regime = "onde progressive (rien n’est réfléchi)"
    elif r > 0.995:
        regime = "onde stationnaire"
    else:
        regime = "onde semi-stationnaire"
    ax.set_title(f"Bord {'libre' if libre else 'fixe'}, r = {r:.2f} : {regime}, amplitude de {1 - r:.2f} à {1 + r:.2f}",
                 fontsize=12)
    ax.legend(loc="lower left", fontsize=8, ncol=3)
    _axes(ax, 2.6)


def _paquet(fig, lam, t, N, larg):
    """Somme de N ondes de nombres d'onde régulièrement espacés dans k0 ± 2σ, avec des poids gaussiens.
    Toutes vont à la même vitesse v = λ/T (milieu non dispersif) : le paquet avance sans se déformer."""
    k0, v = 2 * np.pi / lam, lam
    sigma = larg / 100 * k0
    ecart = np.linspace(-2 * sigma, 2 * sigma, N) if N > 1 else np.zeros(1)
    poids = np.exp(-ecart ** 2 / (2 * sigma ** 2))
    x = np.linspace(-10, 10, 3000)
    xi = (x - (-6.0) - v * t)[None, :]                    # le paquet part de x = −6
    y = (poids[:, None] * np.cos((k0 + ecart)[:, None] * xi)).sum(0) / poids.sum()
    env = np.abs((poids[:, None] * np.exp(1j * ecart[:, None] * xi)).sum(0)) / poids.sum()
    ax, ax2 = fig.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(x, env, "--", color="C1", lw=1.2, label="enveloppe du paquet")
    ax.plot(x, -env, "--", color="C1", lw=1.2)
    ax.plot(x, y, color="C0", lw=1.6, label="somme des ondes")
    if N == 1:
        titre = "Une seule onde : elle remplit tout l’espace, rien n’est localisé"
    else:
        P = 2 * np.pi * (N - 1) / (4 * sigma)             # distance entre deux paquets : 2π / Δk
        if P >= 20:
            titre = f"{N} ondes sommées : un seul paquet dans la fenêtre"
        else:
            titre = f"{N} ondes sommées : le paquet se répète tous les {P:.1f}"
    ax.set_title(titre)
    ax.legend(loc="lower right", fontsize=8)
    _axes(ax, 1.15)
    ax2.vlines(k0 + ecart, 0, poids, color="C0", lw=2)
    ax2.plot(k0 + ecart, poids, "o", color="C0", ms=4)
    ax2.set_xlim(0.25 * k0, 1.75 * k0)
    ax2.set_ylim(0, 1.15)
    ax2.set_xlabel("Nombre d’onde k de chaque onde (rad par unité de longueur)")
    ax2.set_ylabel("Poids")


def fig_onde(fig, mode, lam, t, compo, r, libre, N, larg):
    if mode == PAQ:
        _paquet(fig, lam, t, N, larg)
        return
    ax = fig.subplots()
    if mode == PROG:
        _progressive(ax, lam, t)
    elif mode == STAT:
        _stationnaire(ax, lam, t, compo)
    else:
        _reflexion(ax, lam, t, compo, r, libre)
