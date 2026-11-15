"""Lancers de dés : la somme de n dés comparée à la loi exacte."""
import numpy as np
from kit import *

TITLE = "Lancers de dés"
INTRO = ("On lance n dés à six faces et on additionne les résultats. Les barres montrent la "
         "fréquence observée de chaque somme ; la courbe rouge est la loi exacte, obtenue en "
         "convoluant n fois la loi d’un dé.\n\n"
         "Avec peu de lancers, la simulation fluctue beaucoup. Quand leur nombre augmente, les barres "
         "se calent sur la courbe (loi des grands nombres), et la forme en cloche apparaît dès "
         "que n grandit (théorème central limite).")
RANDOM = True
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


def draw(fig, n_dice, n_rolls):
    ax = fig.subplots()
    lancers = rng.integers(1, 7, size=(n_rolls, n_dice), dtype=np.int8)
    sommes = lancers.sum(axis=1, dtype=np.int32)

    s = np.arange(n_dice, 6 * n_dice + 1)
    obs = np.bincount(sommes - n_dice, minlength=len(s)) / n_rolls * 100
    th = loi_exacte(n_dice) * 100

    ax.bar(s, obs, width=0.9, alpha=0.75, label="simulation")
    ax.plot(s, th, "o-", ms=4, lw=1.6, label="loi exacte")
    if len(s) <= 12:                                   # étiquettes si peu de barres
        for x, h in zip(s, obs):
            ax.text(x, h, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)
    if len(s) <= 25:
        ax.set_xticks(s)
    ax.set_ylim(0, max(obs.max(), th.max()) * 1.15)
    ax.set_xlabel("Somme des dés")
    ax.set_ylabel("Probabilité (%)")
    ax.set_title(f"{n_rolls:,} lancers de {n_dice} dé{'s' if n_dice > 1 else ''}".replace(",", " ")
                 + f"\nmoyenne observée {sommes.mean():.3f}, théorique {3.5 * n_dice:.1f}")
    ax.legend()
