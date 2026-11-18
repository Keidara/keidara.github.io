"""Battements : somme de deux ondes sinusoïdales de fréquences voisines."""
import numpy as np
from kit import *

PARAMS = [
    Slider("f1", "Fréquence f₁", 200, 1000, 440, step=1, unit="Hz"),
    Slider("df", "Écart de fréquence f₂ − f₁", 0.0, 40.0, 6.0, step=0.5, unit="Hz"),
    Slider("a2", "Amplitude de l’onde 2 (relative à l’onde 1)", 0.2, 1.0, 1.0, step=0.05),
]


def fig_zoom(fig, f1, df, a2):
    """Quelques millisecondes : les deux ondes, puis leur somme."""
    f2 = f1 + df
    t = np.linspace(0, 0.02, 2000)
    y1, y2 = np.cos(2 * np.pi * f1 * t), a2 * np.cos(2 * np.pi * f2 * t)
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    ax1.plot(t * 1e3, y1, color="C0", lw=1.6, label=f"onde 1 : {f1} Hz")
    ax1.plot(t * 1e3, y2, color="C1", lw=1.6, label=f"onde 2 : {f2:g} Hz")
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Les deux ondes, sur 20 ms")
    ax1.legend(loc="upper right", fontsize=9, ncol=2)
    ax1.tick_params(labelbottom=False)
    ax2.plot(t * 1e3, y1 + y2, color="C2", lw=1.6)
    ax2.set_ylim(-2.1, 2.1)
    ax2.set_ylabel("Somme")
    ax2.set_xlabel("Temps (ms)")


def fig_battements(fig, f1, df, a2):
    """Plusieurs secondes : la somme et son enveloppe."""
    f2 = f1 + df
    duree = min(3 / df, 4.0) if df > 0 else 1.0
    n = int(min(duree * f2 * 20, 120000))
    t = np.linspace(0, duree, n)
    y = np.cos(2 * np.pi * f1 * t) + a2 * np.cos(2 * np.pi * f2 * t)
    env = np.sqrt(1 + a2 ** 2 + 2 * a2 * np.cos(2 * np.pi * df * t))
    ax = fig.subplots()
    ax.plot(t, y, color="C0", lw=0.5, alpha=0.8, label="somme des deux ondes")
    ax.plot(t, env, color="C1", lw=2, label="enveloppe")
    ax.plot(t, -env, color="C1", lw=2)
    ax.set_xlim(0, duree)
    ax.set_ylim(-2.1, 2.1)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    if df > 0:
        ax.set_title(f"Fréquence de battement |f₂ − f₁| = {df:g} Hz : une période de battement dure {1 / df:.3g} s")
    else:
        ax.set_title("Écart nul : pas de battement")
    ax.legend(loc="upper right", fontsize=9)
