"""Orbites de Kepler : ellipse, aires égales, énergie et troisième loi.
Unités : UA, année, masse du Soleil. Alors GM = 4π² et la période d'une orbite de demi-grand axe a est a^(3/2)."""
import numpy as np
from kit import *

GM = 4 * np.pi ** 2
KMS = 4.74047                       # 1 UA/an en km/s
PRESETS = {"Terre": (1.0, 0.0167), "Mercure": (0.387, 0.2056), "Mars": (1.524, 0.0934),
           "Comète de Halley": (17.83, 0.967)}
PLANETES = [("Mercure", 0.387, 0.241), ("Vénus", 0.723, 0.615), ("Terre", 1.0, 1.0), ("Mars", 1.524, 1.881),
            ("Jupiter", 5.203, 11.86), ("Saturne", 9.537, 29.46), ("Uranus", 19.19, 84.01), ("Neptune", 30.07, 164.8)]

PARAMS = [
    Select("preset", "Orbite", ["Personnalisée", "Terre", "Mercure", "Mars", "Comète de Halley"]),
    Slider("a", "Demi-grand axe a", 0.3, 10.0, 1.5, log=True, unit="UA", actif=("preset", ["Personnalisée"])),
    Slider("e", "Excentricité e", 0.0, 0.95, 0.6, step=0.01, actif=("preset", ["Personnalisée"])),
    Slider("t", "Temps t / T", 0.0, 1.0, 0.0, step=0.005, anim=10),
]


def parametres(preset, a, e):
    return PRESETS.get(preset, (a, e))


def kepler(M, e):
    """Anomalie excentrique E, solution de E − e sin E = M (méthode de Newton)."""
    M = np.asarray(M, float)
    E = np.full_like(M, np.pi) if e > 0.8 else M.copy()
    for _ in range(40):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E


def etat(a, e, f):
    """Anomalie E, position (x, y), distance r et vitesse v (UA/an) à la fraction f de la période.
    Le Soleil est à l'origine, le périhélie sur l'axe x (f = 0)."""
    b = a * np.sqrt(1 - e * e)
    E = kepler(2 * np.pi * np.asarray(f, float), e)
    r = a * (1 - e * np.cos(E))
    return E, a * (np.cos(E) - e), b * np.sin(E), r, np.sqrt(GM * (2 / r - 1 / a))


def fig_orbite(fig, preset, a, e, t):
    a, e = parametres(preset, a, e)
    b, T = a * np.sqrt(1 - e * e), a ** 1.5
    ax = fig.subplots()
    g = np.linspace(0, 2 * np.pi, 400)
    ax.plot(a * (np.cos(g) - e), b * np.sin(g), ":", color="0.5", lw=1)
    for k in range(12):                                         # 12 secteurs balayés en des durées égales T/12
        Ek = kepler(2 * np.pi * np.linspace(k / 12, (k + 1) / 12, 20), e)
        ax.fill(np.r_[0, a * (np.cos(Ek) - e)], np.r_[0, b * np.sin(Ek)], color=f"C{4 + k % 2}", alpha=0.22, lw=0.6)
    E, x, y, r, v = etat(a, e, t)
    x, y, r, v = float(x), float(y), float(r), float(v)
    ax.plot([0, x], [0, y], "-", color="0.5", lw=1)
    d = np.array([-a * np.sin(E), b * np.cos(E)], float)
    d = d / np.hypot(d[0], d[1])
    L = 0.12 * a * min(v / np.sqrt(GM / a), 4)                  # flèche : direction et vitesse
    ax.annotate("", xy=(x + L * d[0], y + L * d[1]), xytext=(x, y), arrowprops=dict(arrowstyle="->", color="C1", lw=2))
    ax.plot([0], [0], "o", color="C3", ms=14, label="Soleil")
    ax.plot([x], [y], "o", color="C0", ms=9, label="planète")
    ax.plot([a * (1 - e)], [0], "x", color="0.4")
    ax.plot([-a * (1 + e)], [0], "x", color="0.4")
    ax.annotate("périhélie", (a * (1 - e), 0), xytext=(0, -16), textcoords="offset points", ha="center", fontsize=9)
    ax.annotate("aphélie", (-a * (1 + e), 0), xytext=(0, -16), textcoords="offset points", ha="center", fontsize=9)
    xc, demi = -a * e, 1.12 * a
    ax.set_xlim(xc - demi, xc + demi)
    ax.set_ylim(-1.15 * max(b, 0.3 * a), 1.15 * max(b, 0.3 * a))
    ax.set_aspect("equal", "datalim")
    ax.set_xlabel("x (UA)")
    ax.set_ylabel("y (UA)")
    ax.set_title(f"a = {a:.3g} UA, e = {e:.3f}, période T = {T:.3g} an\n"
                 f"t = {t * T:.3g} an : r = {r:.3g} UA, v = {v * KMS:.1f} km/s", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)


def fig_energies(fig, preset, a, e):
    a, e = parametres(preset, a, e)
    f = np.linspace(0, 1, 800)
    _, _, _, r, v = etat(a, e, f)
    ec, ep, et = v ** 2 / 2, -GM / r, -GM / (2 * a)
    n = abs(et)
    ax = fig.subplots()
    ax.plot(f, ec / n, color="C0", lw=2, label="énergie cinétique")
    ax.plot(f, ep / n, color="C1", lw=2, label="énergie potentielle")
    ax.plot(f, np.full_like(f, et / n), "--", color="C2", lw=2, label="énergie totale")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Temps t / T (0 : périhélie, 0,5 : aphélie)")
    ax.set_ylabel("Énergie par unité de masse, en |E|")
    ax.set_title("Un échange continu entre énergie cinétique et potentielle")
    ax.legend(loc="center right", fontsize=9)


def fig_troisieme(fig, preset, a):
    a = parametres(preset, a, 0.0)[0]
    ax = fig.subplots()
    A = np.array([p[1] for p in PLANETES])
    T = np.array([p[2] for p in PLANETES])
    g = np.logspace(np.log10(0.25), np.log10(60), 50)
    ax.loglog(g, g ** 1.5, "--", color="C2", lw=1.8, label="T = a^(3/2)")
    ax.loglog(A, T, "o", color="C0", ms=6, label="planètes du système solaire")
    for nom, x, y in PLANETES:
        ax.annotate(nom, (x, y), textcoords="offset points", xytext=(6, -11), fontsize=8)
    ax.loglog([a], [a ** 1.5], "D", color="C1", ms=10, label=f"orbite choisie : T = {a ** 1.5:.3g} an")
    ax.set_xlim(0.25, 60)
    ax.set_ylim(0.12, 400)
    ax.set_xlabel("Demi-grand axe a (UA)")
    ax.set_ylabel("Période T (an)")
    ax.set_title("Troisième loi de Kepler : T² = a³")
    ax.legend(loc="upper left", fontsize=9)
