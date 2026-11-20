"""Marche aléatoire sur un réseau (1D ou 2D) : trajectoires, loi de la position finale, écart quadratique moyen."""
import numpy as np
from kit import *

PARAMS = [
    Select("dim", "Espace", ["Ligne (1D)", "Plan (2D)"]),
    Slider("n_pas", "Nombre de pas", 10, 100_000, 1000, log=True),
    Slider("n_traj", "Trajectoires affichées", 1, 10, 3),
    Slider("n_marcheurs", "Nombre de marcheurs", 10, 100_000, 2000, log=True),
]
rng = np.random.default_rng()


def _plan(dim):
    return dim.startswith("Plan")


def marches(dim, W, n):
    """W marches de n pas. Renvoie x (et y en 2D, sinon None), tableaux (W, n + 1) qui partent de 0.
    1D : un pas de +1 ou −1. 2D : un pas vers l'un des 4 voisins, au hasard."""
    zero = np.zeros((W, 1), np.int32)
    if _plan(dim):
        d = rng.integers(0, 4, (W, n), dtype=np.int8)
        dx = (d == 0).astype(np.int32) - (d == 1).astype(np.int32)
        dy = (d == 2).astype(np.int32) - (d == 3).astype(np.int32)
        return (np.concatenate([zero, np.cumsum(dx, axis=1, dtype=np.int32)], axis=1),
                np.concatenate([zero, np.cumsum(dy, axis=1, dtype=np.int32)], axis=1))
    dx = rng.integers(0, 2, (W, n), dtype=np.int8).astype(np.int32) * 2 - 1
    return np.concatenate([zero, np.cumsum(dx, axis=1, dtype=np.int32)], axis=1), None


def fig_trajectoires(fig, dim, n_pas, n_traj):
    ax = fig.subplots()
    x, y = marches(dim, n_traj, n_pas)
    s = np.sqrt(n_pas)
    if y is not None:
        for i in range(n_traj):
            ax.plot(x[i], y[i], lw=1, color=f"C{i % 6}", alpha=0.9)
            ax.plot([x[i, -1]], [y[i, -1]], "o", color=f"C{i % 6}", ms=6)
        a = np.linspace(0, 2 * np.pi, 400)
        ax.plot(s * np.cos(a), s * np.sin(a), ":", color="0.5", label=r"cercle de rayon $\sqrt{n}$")
        ax.plot([0], [0], "s", color="0.4", ms=7, label="départ")
        ax.set_aspect("equal", "datalim")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
    else:
        t = np.arange(n_pas + 1)
        for i in range(n_traj):
            ax.plot(t, x[i], lw=1, color=f"C{i % 6}")
        ax.plot(t, np.sqrt(t), ":", color="0.5", label=r"$\pm\sqrt{n}$")
        ax.plot(t, -np.sqrt(t), ":", color="0.5")
        ax.set_xlabel("Nombre de pas")
        ax.set_ylabel("Position")
    ax.set_title(f"{n_traj} trajectoire{'s' if n_traj > 1 else ''} de {n_pas:,} pas".replace(",", " "))
    ax.legend(loc="upper left", fontsize=9)


def fig_distribution(fig, dim, n_pas, n_marcheurs):
    ax = fig.subplots()
    n, N = n_pas, n_marcheurs
    if _plan(dim):
        c = rng.multinomial(n, [0.25] * 4, size=N)          # nombre de pas dans chacune des 4 directions
        r = np.hypot(c[:, 0] - c[:, 1], c[:, 2] - c[:, 3])
        ax.hist(r, bins=50, density=True, alpha=0.75, color="C0", label="Distance observée")
        rr = np.linspace(0, max(float(r.max()), 3 * np.sqrt(n)) * 1.05, 400)
        ax.plot(rr, 2 * rr / n * np.exp(-rr ** 2 / n), color="C1", lw=2, label="Loi théorique (Rayleigh)")
        ax.set_xlabel("Distance à l’origine après n pas")
        ax.set_ylabel("Densité de probabilité")
        ax.set_title(f"{N:,} marcheurs, {n:,} pas : ⟨r²⟩ = {np.mean(r ** 2):.4g} (théorie : {n:,})".replace(",", " "))
    else:
        k = rng.binomial(n, 0.5, N)                         # nombre de pas vers la droite
        xs = 2 * np.arange(n + 1) - n                       # position finale correspondante
        cnt = np.bincount(k, minlength=n + 1)
        s = np.sqrt(n)
        garde = np.abs(xs) <= 4 * s + 2
        ax.bar(xs[garde], cnt[garde] / N / 2, width=1.8, alpha=0.75, color="C0", label="Fréquence observée")
        g = np.linspace(-4 * s - 2, 4 * s + 2, 600)
        ax.plot(g, np.exp(-g ** 2 / (2 * n)) / np.sqrt(2 * np.pi * n), color="C1", lw=2, label="Loi normale d’écart-type √n")
        ax.set_xlim(-4 * s - 2, 4 * s + 2)
        x = 2 * k - n
        ax.set_xlabel("Position après n pas")
        ax.set_ylabel("Fréquence ou densité de probabilité")
        ax.set_title(f"{N:,} marcheurs, {n:,} pas : moyenne {x.mean():.3g}, ⟨x²⟩ = {np.mean(x.astype(float) ** 2):.4g} (théorie : {n:,})".replace(",", " "))
    ax.legend(loc="upper right", fontsize=9)


def fig_msd(fig, dim, n_pas, n_marcheurs):
    ax = fig.subplots()
    N = n_marcheurs
    # instants d'observation : grille logarithmique (jusqu'à 100 points), le dernier est n_pas
    t = np.unique(np.logspace(0, np.log10(n_pas), 100).round().astype(int))
    dt = np.diff(np.r_[0, t])
    plan = _plan(dim)
    u = np.zeros(N)
    v = np.zeros(N)
    r2 = []
    for d in dt:
        # déplacement d'une marche à ±1 pendant d pas : 2·Binomiale(d, 1/2) − d
        u += 2 * rng.binomial(d, 0.5, N) - d
        if plan:                                          # en 2D : x+y et x−y sont deux marches 1D indépendantes
            v += 2 * rng.binomial(d, 0.5, N) - d
            r2.append(np.mean(u ** 2 + v ** 2) / 2)
        else:
            r2.append(np.mean(u ** 2))
    ax.loglog(t, r2, color="C0", lw=2, label=f"moyenne sur {N:,} marcheurs".replace(",", " "))
    ax.loglog(t, t, "--", color="C1", lw=2, label=r"Théorie : $\langle r^2\rangle = n$")
    ax.set_xlabel("Nombre de pas $n$")
    ax.set_ylabel(r"$\langle r^2\rangle$" if plan else r"$\langle x^2\rangle$")
    ax.set_title("Le carré de la distance moyenne croît comme le nombre de pas")
    ax.legend(loc="upper left", fontsize=9)
