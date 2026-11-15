"""Monty Hall : garder sa porte ou en changer, sur un grand nombre de parties."""
import numpy as np
from kit import *

TITLE = "Problème de Monty Hall"
INTRO = ("Derrière une porte se trouve un prix, derrière les autres des chèvres. Tu choisis une porte ; "
         "l’animateur, qui sait où est le prix, ouvre alors toutes les autres portes sauf une, en ne "
         "montrant que des chèvres. Tu peux garder ta porte ou prendre celle qui reste.\n\n"
         "Chaque partie est simulée une par une. Garder gagne avec une fréquence proche de 1/n, changer "
         "avec une fréquence proche de (n−1)/n. À droite, on voit les fréquences cumulées se stabiliser "
         "quand le nombre de parties augmente.")
RANDOM = True
PARAMS = [
    Slider("n_doors", "Nombre de portes", 3, 100, 3),
    Slider("n_games", "Nombre de parties", 10, 100_000, 1000, log=True),
]
rng = np.random.default_rng()


def draw(fig, n_doors, n_games):
    prix = rng.integers(0, n_doors, n_games)            # porte du prix
    choix = rng.integers(0, n_doors, n_games)           # première porte choisie
    # Porte laissée fermée par l'animateur : celle du prix si le joueur s'est trompé,
    # sinon une porte à chèvre prise au hasard (décalage aléatoire de 1 à n-1 à partir du prix).
    reste = np.where(choix == prix, (prix + rng.integers(1, n_doors, n_games)) % n_doors, prix)
    garde = choix == prix
    change = reste == prix

    th = np.array([1 / n_doors, (n_doors - 1) / n_doors]) * 100
    freq = np.array([garde.mean(), change.mean()]) * 100

    ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.6]})
    ax1.bar([0, 1], freq, width=0.6, color=["C0", "C1"], alpha=0.75, label="fréquence observée")
    ax1.plot([0, 1], th, "D", color="C3", label="probabilité théorique")
    for x, f in enumerate(freq):
        ax1.text(x, f + 2, f"{f:.1f}%", ha="center", fontsize=10)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["garder", "changer"])
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Parties gagnées (%)")
    ax1.legend(loc="upper left", fontsize=8)

    n = np.arange(1, n_games + 1)
    ax2.plot(n, np.cumsum(garde) / n * 100, color="C0", label="garder")
    ax2.plot(n, np.cumsum(change) / n * 100, color="C1", label="changer")
    ax2.axhline(th[0], color="C0", ls="--", lw=1)
    ax2.axhline(th[1], color="C1", ls="--", lw=1)
    ax2.set_xscale("log")
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("Nombre de parties jouées")
    ax2.set_ylabel("Fréquence cumulée de victoires (%)")
    ax2.legend(loc="center right")
    fig.suptitle(f"{n_games:,} parties avec {n_doors} portes".replace(",", " "))
