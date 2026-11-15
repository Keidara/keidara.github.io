"""Onde progressive et onde stationnaire."""
import numpy as np
from kit import *

TITLE = "Ondes progressive et stationnaire"
INTRO = ("Une onde progressive y = sin(kx − ωt) se déplace sans se déformer, avec la vitesse v = λ/T. "
         "Deux ondes progressives identiques qui vont en sens opposés s’additionnent en une onde "
         "stationnaire y = sin(kx) cos(ωt) : la forme ne se déplace plus, elle oscille sur place.\n\n"
         "Dans l’onde stationnaire, les nœuds (points qui ne bougent jamais) sont espacés de λ/2, et les ventres "
         "(amplitude maximale) tombent au milieu. Le curseur de temps est exprimé en périodes T. "
         "La courbe pointillée gris-bleu montre l’onde à t = 0, pour repérer le déplacement.")
PARAMS = [
    Select("type", "Type d’onde", ["Progressive", "Stationnaire"]),
    Slider("lam", "Longueur d’onde λ", 0.5, 3.0, 1.0, step=0.1),
    Slider("t", "Temps t / T", 0.0, 2.0, 0.0, step=0.02),
    Check("compo", "Montrer les deux ondes qui s’additionnent (stationnaire)", True),
]


def draw(fig, type, lam, t, compo):
    ax = fig.subplots()
    x = np.linspace(0, 4, 800)
    k, w = 2 * np.pi / lam, 2 * np.pi                    # période T = 1

    if type == "Progressive":
        ax.plot(x, np.sin(k * x), ":", color="C2", label="t = 0")
        ax.plot(x, np.sin(k * x - w * t), color="C0", lw=2, label="onde progressive")
        ax.set_title("y = sin(kx − ωt)")
    else:
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
        ax.set_title("y = sin(kx) cos(ωt)")
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
