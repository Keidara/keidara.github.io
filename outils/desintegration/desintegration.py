"""Chaînes de désintégration radioactive : populations exactes et simulation noyau par noyau."""
import numpy as np
from kit import *

# esp : (nom, demi-vie ou None si stable) ; t : plage de temps affichée, dans l'unité de la chaîne
CHAINES = {
    "Un noyau": dict(unite="unités arbitraires (T½ = 1)", t=(0.01, 10),
                     esp=[("noyau radioactif", 1.0), ("noyau fils (stable)", None)]),
    "Chaîne fictive": dict(unite="unités arbitraires", t=(0.01, 30),
                           esp=[("X (T½ = 1)", 1.0), ("Y (T½ = 0,2)", 0.2), ("Z (stable)", None)]),
    "Mo-99 → Tc-99m": dict(unite="heures", t=(0.1, 1000),
                           esp=[("Mo-99", 65.94), ("Tc-99m", 6.007), ("Tc-99", 2.111e5 * 8766), ("Ru-99 (stable)", None)]),
    "Uranium 238": dict(unite="années", t=(1e-7, 2e10),
                        esp=[("U-238", 4.468e9), ("Th-234", 24.10 / 365.25), ("Pa-234m", 1.159 / 525960),
                             ("U-234", 2.455e5), ("descendants (Th-230 … Pb-206)", None)]),
}
PARAMS = [
    Select("chaine", "Chaîne de désintégration", list(CHAINES), "Mo-99 → Tc-99m"),
    Select("affichage", "Affichage", ["Courbes exactes", "Simulation", "Les deux"], "Les deux"),
    Slider("n0", "Nombre initial de noyaux", 10, 1_000_000, 1000, log=True),
    Check("log_y", "Échelle verticale logarithmique", False),
]
rng = np.random.default_rng()
_cache = {}


def _lambdas(nom):
    return np.array([0.0 if th is None else np.log(2) / th for _, th in CHAINES[nom]["esp"]])


def _expm_moins_id(M):
    """exp(M) − I (Taylor + mise au carré). On garde exp(M) − I plutôt que exp(M) : les petites quantités
    (descendants très rares, ou décroissance très lente du parent) ne sont alors pas noyées dans le « 1 »."""
    norme = np.abs(M).sum(axis=0).max()
    s = max(0, int(np.ceil(np.log2(norme))) + 1) if norme > 0 else 0
    B, F, T = M / 2 ** s, np.zeros_like(M), np.eye(len(M))
    for k in range(1, 14):
        T = T @ B / k
        F = F + T
    for _ in range(s):
        F = 2 * F + F @ F                               # (I + F)² − I
    return F


def exact(nom):
    """Fraction de noyaux de chaque espèce au cours du temps, pour un noyau initial (équations de Bateman)."""
    if nom not in _cache:
        lam = _lambdas(nom)
        A = np.diag(-lam)
        A[np.arange(1, len(lam)), np.arange(len(lam) - 1)] = lam[:-1]
        t = np.logspace(*np.log10(CHAINES[nom]["t"]), 240)
        cols = []
        for ti in t:
            c = _expm_moins_id(A * ti)[:, 0]
            c[0] += 1                                   # exp(M) = I + F : la colonne de l'espèce parente
            cols.append(c)
        _cache[nom] = (t, np.array(cols).T)
    return _cache[nom]


def simuler(nom, n0, t):
    """Tire pour chaque noyau l'instant de chacune de ses désintégrations ; compte les espèces aux instants t."""
    lam, naiss, comptes = _lambdas(nom), np.zeros(n0), []
    for l in lam:
        mort = np.full(n0, np.inf) if l == 0 else naiss + rng.exponential(1 / l, n0)
        comptes.append(np.searchsorted(np.sort(naiss), t, "right") - np.searchsorted(np.sort(mort), t, "right"))
        naiss = mort
    return np.array(comptes)


def fig_courbes(fig, chaine, affichage, n0, log_y):
    ax = fig.subplots()
    ch = CHAINES[chaine]
    t, frac = exact(chaine)
    exacte, sim = affichage != "Simulation", affichage != "Courbes exactes"
    if exacte:
        for i, (nom, _) in enumerate(ch["esp"]):
            ax.plot(t, n0 * frac[i], color=f"C{i}", lw=2, label=nom)
    if sim:
        cpt = simuler(chaine, n0, t)
        for i, (nom, _) in enumerate(ch["esp"]):
            ax.plot(t, cpt[i], ".", ms=3.5, color=f"C{i}", alpha=0.85, label=None if exacte else nom)
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(0.5 if sim else max(n0 * 1e-13, 1e-12), n0 * 2)
    else:
        ax.set_ylim(0, n0 * 1.05)
    ax.set_xlabel(f"Temps ({ch['unite']})")
    ax.set_ylabel("Nombre de noyaux")
    ax.set_title(f"{n0:,} noyaux au départ".replace(",", " "))
    ax.legend(loc="center left", fontsize=9)


def fig_activites(fig, chaine, n0):
    ax = fig.subplots()
    ch = CHAINES[chaine]
    t, frac = exact(chaine)
    lam = _lambdas(chaine)
    for i, (nom, _) in enumerate(ch["esp"]):
        if lam[i] > 0:
            ax.plot(t, lam[i] * n0 * frac[i], color=f"C{i}", lw=2, label=nom)
    haut = lam[0] * n0
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(haut * 1e-12, haut * 3)
    ax.set_xlabel(f"Temps ({ch['unite']})")
    ax.set_ylabel("Activité (désintégrations par unité de temps)")
    ax.set_title("Activité de chaque noyau radioactif")
    ax.legend(loc="lower left", fontsize=9)
