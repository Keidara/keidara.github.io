"""Propagation d'une épidémie : modèle SIR/SEIR déterministe, taille finale, simulations aléatoires."""
import numpy as np
from kit import *

PARAMS = [
    Slider("R0", "Nombre de reproduction de base R₀", 0.5, 6.0, 2.5, step=0.1),
    Slider("t_inf", "Durée de la période infectieuse", 2, 21, 7, unit="jours"),
    Slider("t_inc", "Durée d’incubation (0 : modèle sans incubation)", 0, 10, 3, unit="jours"),
    Slider("vacc", "Part de la population immunisée au départ", 0, 90, 0, unit="%"),
    Slider("duree", "Durée affichée", 60, 720, 200, step=10, unit="jours"),
    Slider("n_pop", "Taille de la population (simulations aléatoires)", 200, 20000, 2000, log=True),
]
rng = np.random.default_rng()


def modele(R0, t_inf, t_inc, vacc, duree, i0=1e-4):
    """Intègre S, E, I, R (fractions de la population) par Runge-Kutta, au pas de 0,5 jour."""
    g = 1 / t_inf
    b = R0 * g
    s = 1 / t_inc if t_inc > 0 else 0.0

    def f(y):
        S, E, I, R = y
        inf = b * S * I
        if t_inc > 0:
            return np.array([-inf, inf - s * E, s * E - g * I, g * I])
        return np.array([-inf, 0.0, inf - g * I, g * I])

    dt = 0.5
    n = int(duree / dt)
    y = np.array([1 - vacc / 100 - i0, 0.0, i0, vacc / 100])
    Y = np.empty((n + 1, 4))
    Y[0] = y
    for k in range(n):
        k1 = f(y)
        k2 = f(y + dt / 2 * k1)
        k3 = f(y + dt / 2 * k2)
        k4 = f(y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Y[k + 1] = y
    return np.arange(n + 1) * dt, Y


def taille_finale(R0, v):
    """Part de la population infectée au total, solution de A = (1 − v)(1 − exp(−R₀ A))."""
    R0 = np.asarray(R0, float)
    A = np.full_like(R0, 1 - v)
    for _ in range(400):
        A = (1 - v) * (1 - np.exp(-R0 * A))
    return A


def fig_courbes(fig, R0, t_inf, t_inc, vacc, duree):
    t, Y = modele(R0, t_inf, t_inc, vacc, duree)
    ax = fig.subplots()
    ax.plot(t, 100 * Y[:, 0], color="C0", lw=2, label="sains (susceptibles)")
    if t_inc > 0:
        ax.plot(t, 100 * Y[:, 1], color="C3", lw=2, label="en incubation")
    ax.plot(t, 100 * Y[:, 2], color="C1", lw=2.5, label="infectieux")
    ax.plot(t, 100 * Y[:, 3], color="C2", lw=2, label="guéris ou immunisés")
    pic = int(np.argmax(Y[:, 2]))
    total = 100 * (1 - Y[-1, 0] - vacc / 100)
    ax.set_xlim(0, duree)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Temps (jours)")
    ax.set_ylabel("Part de la population (%)")
    if R0 * (1 - vacc / 100) <= 1:
        ax.set_title(f"R₀ × (1 − immunisés) = {R0 * (1 - vacc / 100):.2f} ≤ 1 : pas d’épidémie")
    else:
        ax.set_title(f"Pic : {100 * Y[pic, 2]:.1f} % d’infectieux au jour {t[pic]:.0f} ; "
                     f"{total:.0f} % de la population infectée à la fin")
    ax.legend(loc="center right", fontsize=9)


def fig_reproduction(fig, R0, vacc):
    ax = fig.subplots()
    r = np.linspace(0.5, 6, 300)
    v = vacc / 100
    ax.plot(r, 100 * taille_finale(r, 0.0), color="0.6", lw=1.5, label="sans immunité au départ")
    ax.plot(r, 100 * taille_finale(r, v), color="C0", lw=2.5, label=f"avec {vacc} % d’immunisés")
    ax.plot([R0], [100 * float(taille_finale(R0, v))], "o", color="C3", ms=10, label=f"R₀ = {R0:.1f}")
    if v < 1:
        ax.axvline(1 / (1 - v), ls=":", color="0.5")
    ax.set_xlim(0.5, 6)
    ax.set_ylim(0, 100)
    ax.set_xlabel("R₀")
    ax.set_ylabel("Population infectée au total (%)")
    if R0 > 1:
        ax.set_title(f"Seuil d’immunité collective pour R₀ = {R0:.1f} : {100 * (1 - 1 / R0):.0f} % d’immunisés")
    else:
        ax.set_title("R₀ ≤ 1 : l’épidémie s’éteint d’elle-même")
    ax.legend(loc="lower right", fontsize=9)


def fig_stochastique(fig, R0, t_inf, t_inc, vacc, duree, n_pop):
    N, runs, dt = int(n_pop), 15, 0.5
    n = int(duree / dt)
    g, b = 1 / t_inf, R0 / t_inf
    s = 1 / t_inc if t_inc > 0 else 0.0
    S0 = max(int(round((1 - vacc / 100) * N)) - 1, 0)
    S = np.full(runs, S0, dtype=np.int64)
    E = np.zeros(runs, dtype=np.int64)
    I = np.ones(runs, dtype=np.int64)                 # un seul infecté au départ
    hist = np.empty((n + 1, runs))
    hist[0] = I
    for k in range(n):
        new_inf = rng.binomial(S, 1 - np.exp(-b * I / N * dt))
        new_rec = rng.binomial(I, 1 - np.exp(-g * dt))
        if t_inc > 0:
            new_i = rng.binomial(E, 1 - np.exp(-s * dt))
            S, E, I = S - new_inf, E + new_inf - new_i, I + new_i - new_rec
        else:
            S, I = S - new_inf, I + new_inf - new_rec
        hist[k + 1] = I
    total = S0 + 1 - S                                 # infectés au total, hors immunisés au départ
    grandes = int(np.sum(total >= 0.05 * N))
    t = np.arange(n + 1) * dt
    ax = fig.subplots()
    ax.plot(t, 100 * hist / N, color="C0", lw=1, alpha=0.5)
    td, Yd = modele(R0, t_inf, t_inc, vacc, duree, i0=1 / N)
    ax.plot(td, 100 * Yd[:, 2], "--", color="C1", lw=2.5, label="modèle déterministe (même départ)")
    reff = R0 * (1 - vacc / 100)
    p = max(0.0, 1 - 1 / reff) if reff > 1 else 0.0
    ax.set_xlim(0, duree)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Temps (jours)")
    ax.set_ylabel("Infectieux (% de la population)")
    ax.set_title(f"{runs} simulations pour {N:,} habitants : {grandes} grandes épidémies (probabilité théorique ≈ {p:.0%})".replace(",", " "))
    ax.legend(loc="upper right", fontsize=9)
