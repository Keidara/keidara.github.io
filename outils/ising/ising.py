"""Modèle d'Ising 2D : algorithme de Metropolis (mises à jour en damier), aimantation et énergie."""
import time
import numpy as np
from collections import deque
from kit import *

PARAMS = [
    Select("depart", "Configuration de départ", ["Aléatoire", "Ordonnée"]),
    Slider("T", "Température T", 0.0, 4.0, 2.0, step=0.05),
    Slider("H", "Champ extérieur h", -1.0, 1.0, 0.0, step=0.05),
]

N = 96
TC = 2 / np.log(1 + np.sqrt(2))                    # température critique d'Onsager ≈ 2,269
rng = np.random.default_rng()
I, J = np.indices((N, N))
MASQUES = [((I + J) % 2) == c for c in (0, 1)]     # damier : deux sous-réseaux indépendants
S = np.ones((N, N), np.int8)
HIST = deque(maxlen=400)                           # (balayages, aimantation, énergie par spin)
DERN = deque(maxlen=5)                             # aimantation après chacun des 5 derniers balayages
bal = 0
PALETTE = np.array([[16, 25, 70, 255], [245, 165, 36, 255]], np.uint8)   # spin bas, spin haut

# La figure « historique » coûte des dizaines à des centaines de ms à tracer, contre environ 1 ms pour un
# balayage : pendant qu'elle se dessine, l'animation est figée. On la rafraîchit donc au plus toutes les
# DELAI_FIG secondes, et on espace davantage ces rafraîchissements quand le tracé est lent (sur un
# ordinateur modeste, par exemple) : il n'occupe jamais plus de 1/FACTEUR_COUT du temps. Une dernière mise
# à jour a lieu à la mise en pause pour que la figure soit à jour.
DELAI_FIG = 0.25
FACTEUR_COUT = 4
_fig = {"t": 0.0, "sale": False, "debut": None, "cout": 0.0}
# Cadence fixe, en balayages par seconde (et non par image) : on cumule le temps écoulé entre deux images
# et on fait autant de balayages entiers que ce temps en permet. Entre 10 et 50, l'animation est fluide.
# DT_MAX évite une rafale de balayages après une pause ou un onglet en arrière-plan.
BALAYAGES_PAR_S = 20
DT_MAX = 0.1
_horloge = {"dernier": None, "credit": 0.0}


def init_ising(depart):
    global S, bal
    S = np.ones((N, N), np.int8) if depart == "Ordonnée" else (rng.integers(0, 2, (N, N)) * 2 - 1).astype(np.int8)
    bal = 0
    HIST.clear()
    DERN.clear()
    _fig.update(t=time.time(), sale=False, debut=None)
    _horloge.update(dernier=None, credit=0.0)
    return ["historique"]


def _balayage(T, H):
    for mk in MASQUES:
        voisins = np.roll(S, 1, 0) + np.roll(S, -1, 0) + np.roll(S, 1, 1) + np.roll(S, -1, 1)
        dE = 2.0 * S * (voisins + H)                # variation d'énergie si on retourne le spin (J = 1)
        accepte = dE <= 0                           # un retournement qui n'augmente pas l'énergie est toujours accepté
        if T > 0:                                   # sinon, avec la probabilité exp(−ΔE/T) ; à T = 0, jamais
            accepte |= rng.random((N, N)) < np.exp(-np.clip(dE, 0, None) / T)
        S[mk & accepte] *= -1


def _mesures(H):
    m = float(S.mean())
    e = -float((S * (np.roll(S, 1, 0) + np.roll(S, 1, 1))).sum()) / N ** 2 - H * m
    return m, e


def frame_ising(lecture, T, H):
    global bal
    maintenant = time.time()
    if _fig["debut"] is not None:                   # la figure vient d'être tracée : on retient ce que ça a coûté
        _fig["cout"] = 0.5 * _fig["cout"] + 0.5 * min(maintenant - _fig["debut"], 0.6)
        _fig["debut"] = None
    dt = 0.0 if _horloge["dernier"] is None else min(maintenant - _horloge["dernier"], DT_MAX)
    _horloge["dernier"] = maintenant
    n = 0
    if lecture:
        _horloge["credit"] += dt * BALAYAGES_PAR_S
        n = int(_horloge["credit"])
        _horloge["credit"] -= n
    else:
        _horloge["credit"] = 0.0
    for _ in range(n):
        _balayage(T, H)
        DERN.append(float(S.mean()))
        bal += 1
    m, e = _mesures(H)
    if n:                                           # une mesure seulement si la grille a changé
        HIST.append((bal, m, e))
        _fig["sale"] = True
    figures = []
    delai = max(DELAI_FIG, FACTEUR_COUT * _fig["cout"])
    if _fig["sale"] and (not lecture or maintenant - _fig["t"] > delai):
        figures = ["historique"]
        _fig.update(t=maintenant, sale=False)
    img = PALETTE[(S > 0).astype(np.intp)]
    texte = f"T = {T:.2f}   Aimantation : {m:+.3f}   Énergie par spin : {e:.3f}   Balayages : {bal}"
    return np.ascontiguousarray(img), dict(taille=N, fini=False, texte=texte, figures=figures)


def fig_historique(fig, T, H):
    _fig["debut"] = time.time()
    # Mise en page fixe : la mise en page automatique coûte presque autant que tout le reste du tracé.
    try:
        fig.set_layout_engine("none")
    except AttributeError:                          # matplotlib < 3.6
        fig.set_constrained_layout(False)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], left=0.075, right=0.985, bottom=0.115, top=0.92,
                          wspace=0.3, hspace=0.12)
    ax_m = fig.add_subplot(gs[0, 0])
    ax_e = fig.add_subplot(gs[1, 0], sharex=ax_m)
    ax_o = fig.add_subplot(gs[:, 1])
    if HIST:
        h = np.array(HIST)
        ax_m.plot(h[:, 0], h[:, 1], color="C0")
        ax_e.plot(h[:, 0], h[:, 2], color="C1")
    ax_m.set_ylim(-1.05, 1.05)
    ax_m.set_ylabel("Aimantation")
    ax_m.set_title("Évolution de l’aimantation et de l’énergie", fontsize=12)
    ax_m.tick_params(labelbottom=False)
    ax_e.set_ylabel("Énergie par spin")
    ax_e.set_xlabel("Balayages")

    # Courbe d'Onsager : deux branches +m et −m (le sens pris par la grille est tiré au hasard)
    t = np.linspace(0.0, 4.1, 600)                  # t = 0 inclus : 2/t vaut alors l'infini, et m = 1
    with np.errstate(all="ignore"):
        onsager = np.where(t < TC, (1 - np.sinh(2 / t) ** -4.0) ** 0.125, 0.0)
    ax_o.axhline(0, color="0.7", lw=0.8)
    alpha = 1.0 if H == 0 else 0.4                  # la théorie ne vaut que pour h = 0
    ax_o.plot(t, onsager, color="C2", lw=2, alpha=alpha, label="Théorie (h = 0)")
    ax_o.plot(t, -onsager, color="C2", lw=2, alpha=alpha)
    # Mesure : aimantation moyennée sur les 5 derniers balayages (un minimum de lissage)
    m_moy = float(np.mean(DERN)) if DERN else float(S.mean())
    nom = "Simulation" if H == 0 else f"Simulation (h = {H:g})"
    ax_o.plot([T], [m_moy], "o", color="C3", ms=9, label=nom)
    if H != 0:
        ax_o.text(0.97, 0.04, "h ≠ 0 : la courbe verte\nne s’applique plus", transform=ax_o.transAxes,
                  ha="right", va="bottom", fontsize=8)
    ax_o.axvline(TC, ls=":", color="0.5")
    ax_o.set_xlabel("Température T")
    ax_o.set_ylabel("Aimantation")
    ax_o.set_title(f"T_c ≈ {TC:.3f}")
    ax_o.set_xlim(-0.1, 4.1)
    ax_o.set_ylim(-1.5, 1.5)
    ax_o.legend(loc="upper right", fontsize=9)
