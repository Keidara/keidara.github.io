"""Lancers de dés : la somme de n dés comparée à la loi exacte."""
import numpy as np
from kit import *

PARAMS = [
    Slider("n_dice", "Nombre de dés", 1, 10, 2),
    Slider("n_rolls", "Nombre de lancers", 1, 1_000_000, 1000, log=True),
]
rng = np.random.default_rng()


def loi_exacte(d):
    """Probabilité de chaque somme d, d+1, ..., 6d."""
    p = np.ones(1)
    for _ in range(d):
        p = np.convolve(p, np.ones(6) / 6)
    return p


def fig_distribution(fig, n_dice, n_rolls):
    ax = fig.subplots()
    lancers = rng.integers(1, 7, size=(n_rolls, n_dice), dtype=np.int8)
    sommes = lancers.sum(axis=1, dtype=np.int32)

    s = np.arange(n_dice, 6 * n_dice + 1)
    freq = np.bincount(sommes - n_dice, minlength=len(s)) / n_rolls * 100   # occurrences / lancers
    th = loi_exacte(n_dice) * 100

    ax.bar(s, freq, width=0.9, alpha=0.75, color="C0", label="Fréquence du tirage")
    ax.plot(s, th, "o-", ms=4, lw=1.6, color="C1", label="Loi de probabilité théorique")
    if len(s) <= 12:                                   # étiquettes si peu de barres
        for x, h in zip(s, freq):
            ax.text(x, h, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)
    if len(s) <= 25:
        ax.set_xticks(s)
    ax.set_ylim(0, max(freq.max(), th.max()) * 1.15)
    ax.set_xlabel("Somme des dés")
    ax.set_ylabel("Fréquence du tirage ou probabilité (%)")
    ax.set_title(f"{n_rolls:,} lancers de {n_dice} dé{'s' if n_dice > 1 else ''}".replace(",", " ")
                 + f"\nMoyenne observée {sommes.mean():.3f}, théorique {3.5 * n_dice:.1f}" + f"\nÉcart-type {sommes.std():.1f}")
    ax.legend()
