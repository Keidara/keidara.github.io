"""Relativité restreinte : dilatation du temps, contraction des longueurs, énergie, composition des vitesses."""
import numpy as np
from kit import *

PARAMS = [
    Slider("beta", "Vitesse v / c", 0.0, 0.999, 0.5, step=0.001),
    Slider("u", "Vitesse du projectile u / c", 0.0, 0.99, 0.5, step=0.01),
]


def gamma(b):
    return 1 / np.sqrt(1 - np.asarray(b) ** 2)


def fig_horloge(fig, beta):
    """Horloge à photons : un photon rebondit entre deux miroirs distants de L (c = 1, L = 1).
    Les deux horloges sont dessinées dans le même repère, donc à la même échelle."""
    g = float(gamma(beta))
    ax = fig.subplots()

    def miroirs(x0, x1):
        for y in (0, 1):
            ax.plot([x0, x1], [y, y], lw=5, color="0.5", solid_capstyle="butt")

    def fleche(a, b):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="C3", lw=2))

    miroirs(-0.6, 0.6)                                   # horloge au repos
    fleche((-0.1, 0), (-0.1, 1))
    fleche((0.1, 1), (0.1, 0))
    ax.text(0, 1.15, "Dans le vaisseau\ntrajet 2L", ha="center", va="bottom")
    x0, X = 2.0, 2 * beta * g                            # même horloge, vue du laboratoire
    miroirs(x0 - 0.6, x0 + X + 0.6)
    fleche((x0, 0), (x0 + beta * g, 1))
    fleche((x0 + beta * g, 1), (x0 + X, 0))
    ax.text(x0 + X / 2, 1.15, f"Vu du laboratoire (à l’échelle)\ntrajet 2γL = {2 * g:.2f} L", ha="center", va="bottom")
    ax.set_xlim(-1, x0 + X + 1)
    ax.set_ylim(-0.3, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_facteurs(fig, beta):
    x = np.linspace(0, 0.999, 500)
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    ax1.plot(x, gamma(x), color="C0", lw=2)
    ax1.set_yscale("log")
    ax1.set_ylim(1, 25)
    ax1.set_ylabel("γ")
    ax1.set_title(f"Dilatation du temps : γ = {float(gamma(beta)):.3f}")
    ax2.plot(x, 1 / gamma(x), color="C2", lw=2)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("L / L₀")
    ax2.set_xlabel("Vitesse v / c")
    ax2.set_title(f"Contraction des longueurs : L / L₀ = {float(1 / gamma(beta)):.3f}")
    for ax, y in ((ax1, float(gamma(beta))), (ax2, float(1 / gamma(beta)))):
        ax.axvline(beta, ls="--", color="C1", lw=1)
        ax.plot([beta], [y], "o", color="C1")


def fig_energie(fig, beta):
    x = np.linspace(0.001, 0.999, 800)
    ax = fig.subplots()
    ax.semilogy(x, gamma(x) - 1, color="C0", lw=2, label="relativiste : (γ − 1) mc²")
    ax.semilogy(x, 0.5 * x ** 2, "--", color="C1", lw=2, label="classique : ½ m v²")
    ax.set_ylim(1e-6, 30)
    ax.set_xlabel("Vitesse v / c")
    ax.set_ylabel("Énergie cinétique (en mc²)")
    if beta > 0.001:
        rel, cl = float(gamma(beta)) - 1, 0.5 * beta ** 2
        ax.plot([beta], [rel], "o", color="C0")
        ax.plot([beta], [cl], "o", color="C1")
        ax.set_title(f"E_c = {rel:.3g} mc², la formule classique donne {cl:.3g} mc² (erreur {1 - cl / rel:.1%})")
    ax.legend(loc="upper left")


def fig_composition(fig, beta, u):
    x = np.linspace(0, 0.999, 400)
    ax = fig.subplots()
    ax.plot(x, (u + x) / (1 + u * x), color="C0", lw=2, label="relativiste : (u + v) / (1 + uv/c²)")
    ax.plot(x, u + x, "--", color="C1", lw=2, label="classique : u + v")
    ax.axhline(1, ls=":", color="0.5", label="vitesse de la lumière")
    w = (u + beta) / (1 + u * beta)
    ax.plot([beta, beta], [w, u + beta], "o", color="C3")
    ax.set_ylim(0, 2)
    ax.set_xlabel("Vitesse du vaisseau v / c")
    ax.set_ylabel("Vitesse du projectile vue du laboratoire / c")
    ax.set_title(f"Projectile à {u:.2f} c tiré depuis un vaisseau à {beta:.3f} c : {w:.3f} c (classique : {u + beta:.3f} c)")
    ax.legend(loc="lower right")
