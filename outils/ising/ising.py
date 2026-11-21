"""Modèle d'Ising 2D : algorithme de Metropolis (mises à jour en damier), aimantation et énergie."""
import time
import numpy as np
from collections import deque
from kit import *

PARAMS = [
    Select("depart", "Configuration de départ", ["Aléatoire", "Ordonnée"]),
    Slider("T", "Température T", 0.5, 5.0, 2.0, step=0.05),
    Slider("H", "Champ extérieur h", -1.0, 1.0, 0.0, step=0.05)
]

N = 96
TC = 2 / np.log(1 + np.sqrt(2))                    # température critique d'Onsager ≈ 2,269
rng = np.random.default_rng()
I, J = np.indices((N, N))
MASQUES = [((I + J) % 2) == c for c in (0, 1)]     # damier : deux sous-réseaux indépendants
S = np.ones((N, N), np.int8)
HIST = deque(maxlen=400)                           # (balayages, aimantation, énergie par spin, température)
bal = 0
PALETTE = np.array([[16, 25, 70, 255], [245, 165, 36, 255]], np.uint8)   # spin bas, spin haut

# La figure « historique » coûte plus cher à tracer qu'une image de la grille : on la rafraîchit au plus
# toutes les DELAI_FIG secondes, et une dernière fois à la mise en pause pour qu'elle soit à jour.
DELAI_FIG = 0.25
_fig = {"t": 0.0, "sale": False}
# La vitesse est en balayages par seconde (et non par image) : on cumule le temps écoulé entre deux images
# et on fait autant de balayages entiers que ce temps en permet. DT_MAX évite une rafale de balayages après
# une pause ou un onglet en arrière-plan.
DT_MAX = 0.1
_horloge = {"dernier": None, "credit": 0.0}


def init_ising(depart):
    global S, bal
    S = np.ones((N, N), np.int8) if depart == "Ordonnée" else (rng.integers(0, 2, (N, N)) * 2 - 1).astype(np.int8)
    bal = 0
    HIST.clear()
    _fig.update(t=time.time(), sale=False)
    _horloge.update(dernier=None, credit=0.0)
    return ["historique"]


def _balayage(T, H):
    for mk in MASQUES:
        voisins = np.roll(S, 1, 0) + np.roll(S, -1, 0) + np.roll(S, 1, 1) + np.roll(S, -1, 1)
        dE = 2.0 * S * (voisins + H)                # variation d'énergie si on retourne le spin (J = 1)
        accepte = (dE <= 0) | (rng.random((N, N)) < np.exp(-np.clip(dE, 0, None) / T))
        S[mk & accepte] *= -1


def _mesures(H):
    m = float(S.mean())
    e = -float((S * (np.roll(S, 1, 0) + np.roll(S, 1, 1))).sum()) / N ** 2 - H * m
    return m, e


def frame_ising(lecture, T, H, vitesse):
    global bal
    maintenant = time.time()
    dt = 0.0 if _horloge["dernier"] is None else min(maintenant - _horloge["dernier"], DT_MAX)
    _horloge["dernier"] = maintenant
    n = 0
    if lecture:
        _horloge["credit"] += dt * vitesse
        n = int(_horloge["credit"])
        _horloge["credit"] -= n
    else:
        _horloge["credit"] = 0.0
    for _ in range(n):
        _balayage(T, H)
        bal += 1
    m, e = _mesures(H)
    if n:                                           # une mesure seulement si la grille a changé
        HIST.append((bal, m, e, T))
        _fig["sale"] = True
    figures = []
    if _fig["sale"] and (not lecture or maintenant - _fig["t"] > DELAI_FIG):
        figures = ["historique"]
        _fig.update(t=maintenant, sale=False)
    img = PALETTE[(S > 0).astype(np.intp)]
    texte = f"T = {T:.2f}   Aimantation : {m:+.3f}   Énergie par spin : {e:.3f}   Balayages : {bal}"
    return np.ascontiguousarray(img), dict(taille=N, fini=False, texte=texte, figures=figures)


def fig_historique(fig, T):
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    ax_m = fig.add_subplot(gs[0, 0])
    ax_e = fig.add_subplot(gs[1, 0], sharex=ax_m)
    ax_o = fig.add_subplot(gs[:, 1])
    if HIST:
        h = np.array(HIST)
        ax_m.plot(h[:, 0], h[:, 1], color="C0")
        ax_e.plot(h[:, 0], h[:, 2], color="C1")
    ax_m.set_ylim(-1.05, 1.05)
    ax_m.set_ylabel("aimantation")
    ax_m.set_title("Évolution de l’aimantation et de l’énergie")
    ax_m.tick_params(labelbottom=False)
    ax_e.set_ylabel("énergie par spin")
    ax_e.set_xlabel("balayages")

    # Courbe d'Onsager : deux branches +m et −m (le sens pris par la grille est tiré au hasard)
    t = np.linspace(0.3, 5, 600)
    with np.errstate(all="ignore"):
        onsager = np.where(t < TC, (1 - np.sinh(2 / t) ** -4.0) ** 0.125, 0.0)
    ax_o.axhline(0, color="0.7", lw=0.8)
    ax_o.plot(t, onsager, color="C2", lw=2, label="théorie (h = 0)")
    ax_o.plot(t, -onsager, color="C2", lw=2)
    # Mesure : aimantation signée, moyennée sur les dernières images prises à la température actuelle
    recents = [v[1] for v in list(HIST)[-60:] if abs(v[3] - T) < 1e-9]
    if len(recents) > 5:
        ax_o.plot([T], [np.mean(recents)], "o", color="C3", ms=9, label="simulation")
    ax_o.axvline(TC, ls=":", color="0.5")
    ax_o.set_xlabel("température T")
    ax_o.set_ylabel("aimantation")
    ax_o.set_title(f"T_c ≈ {TC:.3f}")
    ax_o.set_ylim(-1.05, 1.05)
    ax_o.legend(loc="upper right", fontsize=9)
