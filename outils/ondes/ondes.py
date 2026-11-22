"""Ondes : progressive, stationnaire, réflexion sur un bord, somme d'ondes de même amplitude."""
import numpy as np
from kit import *

PROG, STAT, REFL, PAQ = "Progressive", "Stationnaire", "Réflexion", "Paquet d’ondes"

PARAMS = [
    Select("mode", "Mode", [PROG, STAT, REFL, PAQ]),
    Slider("lam", r"Longueur d’onde \(\lambda\)", 0.5, 3.0, 1.0, step=0.1,
           actif=("mode", [PROG, STAT, REFL]), cache=True),
    Slider("r", r"Coefficient de réflexion \(r\) (amplitude réfléchie / incidente)", 0.0, 1.0, 0.7, step=0.05,
           actif=("mode", [REFL]), cache=True),
    Check("libre", "Bord libre (sinon bord fixe)", False, actif=("mode", [REFL]), cache=True),
    Slider("N", r"Nombre d’ondes sommées \(N\)", 1, 30, 5, actif=("mode", [PAQ]), cache=True),
    Slider("n0", r"Rang de la première onde \(n_0\)", 1, 30, 10, actif=("mode", [PAQ]), cache=True),
    Check("comps", "Montrer chaque onde, décalée verticalement", True, actif=("mode", [PAQ]), cache=True),
    Slider("t", r"Temps \(t / T\)", 0.0, 2.0, 0.0, step=0.02, anim=8),
    Check("compo", "Montrer les deux ondes qui s’additionnent", False, actif=("mode", [STAT, REFL]), cache=True),
    Check("montrer_env", "Afficher l’enveloppe", True, actif=("mode", [STAT, REFL, PAQ]), cache=True),
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


def _stationnaire(ax, lam, t, compo, montrer_env):
    x = np.linspace(0, 4, 800)
    k, w = 2 * np.pi / lam, 2 * np.pi
    aller = 0.5 * np.sin(k * x - w * t)
    retour = 0.5 * np.sin(k * x + w * t)
    if compo:
        ax.plot(x, aller, "--", color="C2", lw=1, label="onde vers la droite")
        ax.plot(x, retour, "--", color="C3", lw=1, label="onde vers la gauche")
    if montrer_env:
        ax.plot(x, np.abs(np.sin(k * x)), ":", color="C1", lw=1)
        ax.plot(x, -np.abs(np.sin(k * x)), ":", color="C1", lw=1, label="enveloppe")
    ax.plot(x, aller + retour, color="C0", lw=2, label="somme (stationnaire)")
    noeuds = np.arange(0, 4 + 1e-9, lam / 2)
    ax.plot(noeuds, np.zeros_like(noeuds), "o", color="C1", ms=6, label="nœuds")
    ax.set_title(f"t = {t:.2f} T : la forme oscille sur place, entre les nœuds")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    _axes(ax, 1.15)


def _reflexion(ax, lam, t, compo, r, libre, montrer_env):
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
    if montrer_env:
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


def _paquet(fig, t, N, n0, comps, montrer_env):
    """Somme de N ondes de même amplitude 1/N et de rangs n0, n0+1, ... (la période spatiale Λ de la fondamentale
    et sa période T sont les unités : l'onde de rang n a la longueur d'onde Λ/n, et toutes vont à la vitesse Λ/T)."""
    x = np.linspace(-1, 1, 4000)
    xi = x - t                                            # x − vt, avec v = 1
    rangs = np.arange(n0, n0 + N)
    ondes = np.cos(2 * np.pi * rangs[:, None] * xi[None, :])
    y = ondes.sum(axis=0) / N
    s = np.sin(np.pi * xi)
    with np.errstate(all="ignore"):
        env = np.where(np.abs(s) < 1e-9, 1.0, np.abs(np.sin(N * np.pi * xi) / (N * s)))
    centres = [c for c in np.arange(-3, 4) + t if -1 <= c <= 1]     # là où toutes les ondes sont en phase
    if comps:
        ax, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1.6]})
    else:
        ax, ax2 = fig.subplots(), None
    if montrer_env:
        ax.plot(x, env, "--", color="C1", lw=1.2, label="enveloppe")
        ax.plot(x, -env, "--", color="C1", lw=1.2)
    ax.plot(x, y, color="C0", lw=1.4, label="somme des ondes")
    for c in centres:
        ax.axvline(c, ls=":", color="0.5", lw=1)
    if N == 1:
        ax.set_title("Une seule onde : elle remplit tout l’espace, rien n’est localisé")
    else:
        ax.set_title(f"{N} ondes de même amplitude : pics espacés de Λ, de largeur de l’ordre de Λ/{N}")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlim(-1, 1)
    ax.set_ylabel("Amplitude de la somme")
    if ax2 is None:
        ax.set_xlabel("Position x (en périodes spatiales Λ)")
        return
    ax.tick_params(labelbottom=False)
    for i in range(N):
        ax2.plot(x, i + 0.4 * ondes[i], color="C0", lw=1.0 if N <= 12 else 0.7)
    for c in centres:
        ax2.axvline(c, ls=":", color="0.5", lw=1)
    pas = int(np.ceil(N / 8))
    ax2.set_yticks(range(0, N, pas))
    ax2.set_yticklabels([str(n) for n in rangs[::pas]])
    ax2.set_ylim(-0.6, N - 0.4)
    ax2.set_ylabel("Rang n de chaque onde")
    ax2.set_xlabel("Position x (en périodes spatiales Λ)")
    ax2.set_title("Chaque onde, décalée verticalement : toutes sont en phase aux pointillés", fontsize=11)


def fig_onde(fig, mode, lam, t, compo, r, libre, N, n0, comps, montrer_env):
    if mode == PAQ:
        _paquet(fig, t, N, n0, comps, montrer_env)
        return
    ax = fig.subplots()
    if mode == PROG:
        _progressive(ax, lam, t)
    elif mode == STAT:
        _stationnaire(ax, lam, t, compo, montrer_env)
    else:
        _reflexion(ax, lam, t, compo, r, libre, montrer_env)
