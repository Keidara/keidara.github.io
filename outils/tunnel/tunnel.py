"""Effet tunnel 1D : paquet d'onde contre une barrière de potentiel rectangulaire (split-step FFT, ħ = m = 1)."""
import numpy as np
from kit import *

PARAMS = [
    Slider("E", "Énergie du paquet E", 0.1, 3.0, 0.5, step=0.05),
    Slider("V0", "Hauteur de la barrière V₀", 0.5, 3.0, 1.0, step=0.05),
    Slider("a", "Largeur de la barrière a", 0.5, 6.0, 2.0, step=0.1),
    Slider("t", "Temps", 0.0, 50.0, 0.0, step=0.1, anim=12),
]

L, N = 160.0, 1024                    # domaine [-L/2, L/2]
DT, NPAS, PAR = 0.05, 1000, 10        # pas de temps, nombre de pas, une image enregistrée tous les PAR pas
T_TOT = NPAS * DT
SIGMA, X0 = 4.0, -25.0                # largeur et position initiale du paquet
dx = L / N
x = (np.arange(N) - N // 2) * dx
k = 2 * np.pi * np.fft.fftfreq(N, dx)
KIN = np.exp(-0.5j * k ** 2 * DT)
_s = np.clip(np.abs(x) - (L / 2 - 15), 0, None) / 15
W = 2.0 * _s ** 2                     # bords absorbants
_cache = {}


def transmission(E, V0, a):
    """Coefficient de transmission exact d'une onde plane (barrière rectangulaire)."""
    E = np.asarray(E, float)
    if V0 <= 0:
        return np.ones_like(E)
    out = np.empty_like(E)
    bas = E < V0
    kap = np.sqrt(2 * np.clip(V0 - E, 1e-12, None))
    kk = np.sqrt(2 * np.clip(E - V0, 1e-12, None))
    out[bas] = 1 / (1 + V0 ** 2 * np.sinh(kap[bas] * a) ** 2 / (4 * E[bas] * (V0 - E[bas])))
    out[~bas] = 1 / (1 + V0 ** 2 * np.sin(kk[~bas] * a) ** 2 / (4 * E[~bas] * (E[~bas] - V0) + 1e-12))
    return out


def evolution(E, V0, a):
    """Calcule (et garde en mémoire) l'évolution du paquet : images de |ψ|² et transmission mesurée."""
    clef = (round(E, 3), round(V0, 3), round(a, 3))
    if clef not in _cache:
        if len(_cache) > 3:
            _cache.clear()
        V = np.where(np.abs(x) <= a / 2, V0, 0.0)
        half = np.exp((-1j * V - W) * DT / 2)
        k0 = np.sqrt(2 * E)
        psi = np.exp(-(x - X0) ** 2 / (4 * SIGMA ** 2) + 1j * k0 * x)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
        ip = int(np.argmin(np.abs(x - (a / 2 + 6))))                # point de mesure du flux transmis
        images, flux = [np.abs(psi) ** 2], 0.0
        for i in range(NPAS):
            psi = psi * half
            psi = np.fft.ifft(np.fft.fft(psi) * KIN)
            psi *= half
            j = np.imag(np.conj(psi[ip]) * (psi[ip + 1] - psi[ip - 1]) / (2 * dx))
            flux += j * DT
            if (i + 1) % PAR == 0:
                images.append(np.abs(psi) ** 2)
        _cache[clef] = (np.array(images), float(np.clip(flux, 0, 1)))
    return _cache[clef]


def fig_paquet(fig, E, V0, a, t):
    ax = fig.subplots()
    images, T_mes = evolution(E, V0, a)
    p = images[int(round(t / T_TOT * (len(images) - 1)))]
    gauche, droite = p[x < -a / 2].sum() * dx, p[x > a / 2].sum() * dx
    ax.axvspan(-a / 2, a / 2, color="0.5", alpha=0.4, label=f"barrière (V₀ = {V0:g})")
    ax.fill_between(x, p, color="C0", alpha=0.35)
    ax.plot(x, p, color="C0", lw=1.8, label="densité de probabilité |ψ|²")
    ax.set_xlim(-60, 60)
    ax.set_ylim(0, 1.08 * images.max())                 # échelle fixe pendant toute l'animation
    ax.set_xlabel("x")
    ax.set_ylabel("|ψ|²")
    ax.set_title(f"t = {t:.1f} — à gauche de la barrière : {gauche:.0%}, à droite : {droite:.0%}")
    ax.legend(loc="upper right")


def fig_transmission(fig, E, V0, a):
    ax = fig.subplots()
    _, T_mes = evolution(E, V0, a)
    e = np.linspace(0.02, 3.0, 800)
    ax.plot(e / V0, transmission(e, V0, a), color="C1", lw=2, label="transmission d’une onde plane")
    ax.plot([E / V0], [T_mes], "o", color="C0", ms=9, label="paquet simulé")
    ax.axvline(1, ls=":", color="0.5")
    ax.set_xlim(0, 3.0 / V0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("E / V₀")
    ax.set_ylabel("Probabilité de transmission")
    ax.set_title(f"E = {E:g}, V₀ = {V0:g}, a = {a:g} : transmission {float(transmission(np.array([E]), V0, a)[0]):.1%} (onde plane), {T_mes:.1%} (paquet)")
    ax.legend(loc="lower right")
