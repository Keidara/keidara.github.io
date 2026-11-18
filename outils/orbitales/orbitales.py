"""Orbitales de l'atome d'hydrogène : coupe de la densité de probabilité et densité radiale (unités atomiques, a₀ = 1)."""
import numpy as np
from math import factorial
from kit import *

PARAMS = [
    Slider("n", "Nombre quantique principal n", 1, 5, 3),
    Slider("l", "Nombre quantique orbital l", 0, 4, 1),
    Slider("m", "Nombre quantique magnétique m", -4, 4, 0),
]
LETTRES = "spdfg"


def laguerre(k, alpha, x):
    """Polynôme de Laguerre généralisé L_k^(alpha)(x), par récurrence."""
    L0 = np.ones_like(x)
    if k == 0:
        return L0
    L1 = 1 + alpha - x
    for j in range(1, k):
        L0, L1 = L1, ((2 * j + 1 + alpha - x) * L1 - (j + alpha) * L0) / (j + 1)
    return L1


def radiale(n, l, r):
    """Fonction d'onde radiale R_nl(r), normalisée : ∫ R² r² dr = 1."""
    rho = 2 * r / n
    norme = np.sqrt((2 / n) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
    return norme * np.exp(-rho / 2) * rho ** l * laguerre(n - l - 1, 2 * l + 1, rho)


def legendre(l, m, x):
    """Fonction de Legendre associée P_l^m(x), m ≥ 0."""
    pmm = np.ones_like(x)
    if m > 0:
        somx2, fact = np.sqrt((1 - x) * (1 + x)), 1.0
        for _ in range(m):
            pmm = -pmm * fact * somx2
            fact += 2
    if l == m:
        return pmm
    pmm1 = x * (2 * m + 1) * pmm
    for ll in range(m + 2, l + 1):
        pmm, pmm1 = pmm1, (x * (2 * ll - 1) * pmm1 - (ll + m - 1) * pmm) / (ll - m)
    return pmm1


def angulaire2(l, m, cos_t):
    """|Y_l^m|² (ne dépend pas de l'angle φ)."""
    m = abs(m)
    return (2 * l + 1) / (4 * np.pi) * factorial(l - m) / factorial(l + m) * legendre(l, m, cos_t) ** 2


def _nombres(n, l, m):
    l = min(l, n - 1)
    return n, l, max(-l, min(l, m))


def fig_coupe(fig, n, l, m):
    n, l, m = _nombres(n, l, m)
    ax = fig.subplots()
    demi = 2.5 * n * n + 4
    g = np.linspace(-demi, demi, 500)
    X, Z = np.meshgrid(g, g)
    r = np.hypot(X, Z)
    dens = radiale(n, l, r) ** 2 * angulaire2(l, m, np.where(r > 0, Z / np.where(r > 0, r, 1), 1.0))
    ax.imshow(dens[::-1] ** 0.5, extent=[-demi, demi, -demi, demi], cmap="magma", vmin=0)
    ax.set_xlabel("x (a₀)")
    ax.set_ylabel("z (a₀)")
    ax.set_title(f"n = {n}, l = {l}, m = {m} : orbitale {n}{LETTRES[l]} — coupe dans un plan contenant l’axe z")


def fig_radial(fig, n, l):
    n, l = n, min(l, n - 1)
    ax = fig.subplots()
    rmax = 3.2 * n * n + 6
    r = np.linspace(0, rmax, 1500)
    P = r ** 2 * radiale(n, l, r) ** 2
    moyen = 0.5 * (3 * n * n - l * (l + 1))
    ax.fill_between(r, P, color="C0", alpha=0.3)
    ax.plot(r, P, color="C0", lw=2)
    ax.axvline(moyen, ls="--", color="C1", label=f"⟨r⟩ = {moyen:g} a₀")
    ax.set_xlim(0, rmax)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("r (a₀)")
    ax.set_ylabel("densité radiale r² R²")
    ax.set_title(f"Probabilité de trouver l’électron à la distance r ({n - l - 1} nœud{'s' if n - l - 1 > 1 else ''} radia{'ux' if n - l - 1 > 1 else 'l'})")
    ax.legend(loc="upper right")
