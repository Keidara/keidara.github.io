"""Électron dans un puits de potentiel infini : états propres, niveaux d'énergie, superposition de deux états."""
import numpy as np
from kit import *

PARAMS = [
    Select("mode", "État", ["État propre", "Superposition"]),
    Slider("L", "Largeur du puits L", 0.2, 2.0, 1.0, step=0.05, unit="nm"),
    Slider("n", "Niveau n", 1, 8, 2),
    Slider("n2", "Second niveau n′", 1, 8, 3, actif=("mode", ["Superposition"])),
    Slider("t", "Temps t / T", 0.0, 1.0, 0.0, step=0.005, anim=8, actif=("mode", ["Superposition"])),
]

E1 = 0.37605          # h²/8m pour un électron, en eV·nm² : E_n = E1 n² / L²
H_EVFS = 4.135667696  # constante de Planck, en eV·fs
HC = 1239.84198       # h·c en eV·nm


def energie(k, L):
    return E1 * k ** 2 / L ** 2


def etat(mode, L, n, n2, t, x):
    """Fonction d'onde ψ(x, t) (complexe) et énergie de battement ΔE (None pour un état propre)."""
    psi = lambda k: np.sqrt(2 / L) * np.sin(k * np.pi * x / L)
    if mode == "Superposition" and n2 != n:
        dE = abs(energie(n2, L) - energie(n, L))
        ph = lambda k: np.exp(-2j * np.pi * t * energie(k, L) / dE)      # t est mesuré en périodes T = h/ΔE
        return (psi(n) * ph(n) + psi(n2) * ph(n2)) / np.sqrt(2), dE
    return psi(n).astype(complex), None


def fig_fonction(fig, mode, L, n, n2, t):
    x = np.linspace(0, L, 600)
    psi, dE = etat(mode, L, n, n2, t, x)
    dens = np.abs(psi) ** 2
    xm = float(np.sum(x * dens) * (x[1] - x[0]))                          # position moyenne
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    ax1.plot(x, psi.real, color="C0", lw=2, label="partie réelle")
    if dE is not None:
        ax1.plot(x, psi.imag, "--", color="C1", lw=1.5, label="partie imaginaire")
        ax1.legend(loc="upper right", fontsize=9)
    ax1.axhline(0, color="0.5", lw=0.8)
    ax1.set_ylabel("ψ (nm⁻¹ᐟ²)")
    if dE is None:
        noeuds = n - 1
        ax1.set_title(f"État propre n = {n} : E = {energie(n, L):.3g} eV, {noeuds} nœud{'s' if noeuds > 1 else ''}")
    else:
        ax1.set_title(f"Superposition des niveaux {n} et {n2} : période du battement T = {H_EVFS / dE:.3g} fs")
    ax2.fill_between(x, dens, color="C2", alpha=0.35)
    ax2.plot(x, dens, color="C2", lw=2)
    ax2.axvline(xm, ls="--", color="C3", label=f"position moyenne : {xm:.3f} nm")
    ax2.set_ylabel("|ψ|² (nm⁻¹)")
    ax2.set_xlabel("Position x (nm)")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper right", fontsize=9)
    for ax in (ax1, ax2):
        ax.axvline(0, color="0.4", lw=3)
        ax.axvline(L, color="0.4", lw=3)
        ax.set_xlim(-0.08 * L, 1.08 * L)


def fig_niveaux(fig, mode, L, n, n2):
    ax = fig.subplots()
    for k in range(1, 9):
        sel = k == n or (mode == "Superposition" and k == n2)
        ax.hlines(energie(k, L), 0, 1, color="C0" if sel else "0.6", lw=3 if sel else 1.5)
        ax.text(1.03, energie(k, L), f"n = {k} : {energie(k, L):.3g} eV", va="center", fontsize=9)
    if mode == "Superposition" and n2 != n:
        e1, e2 = energie(n, L), energie(n2, L)
        dE = abs(e2 - e1)
        ax.annotate("", xy=(0.5, max(e1, e2)), xytext=(0.5, min(e1, e2)), arrowprops=dict(arrowstyle="<->", color="C1", lw=2))
        ax.text(0.45, (e1 + e2) / 2, f"ΔE = {dE:.3g} eV\nphoton de {HC / dE:.3g} nm", ha="right", va="center", color="C1", fontsize=10)
    ax.set_xlim(0, 1.6)
    ax.set_xticks([])
    ax.set_ylabel("Énergie (eV)")
    ax.set_title("Niveaux d’énergie : E_n = E₁ n²")
