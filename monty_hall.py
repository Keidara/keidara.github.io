"""Monty Hall : à jouer (clique sur les portes) ou à simuler sur un grand nombre de parties."""
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from kit import *

TITLE = "Problème de Monty Hall"
INTRO = ("Derrière une porte se trouve un prix, derrière les autres des chèvres. Tu choisis une porte ; "
         "l’animateur, qui sait où est le prix, ouvre alors toutes les autres portes sauf une, en ne "
         "montrant que des chèvres. Tu peux garder ta porte ou prendre celle qui reste.\n\n"
         "Mode Jouer : clique sur une porte, puis sur Garder ou Changer. Le score cumulé de chaque stratégie "
         "s’affiche en bas. Mode Simuler : chaque partie est jouée par l’ordinateur. Garder gagne avec une "
         "fréquence proche de 1/n, changer avec une fréquence proche de (n−1)/n.")
RANDOM = True
PARAMS = [
    Select("mode", "Mode", ["Jouer", "Simuler"], "Jouer"),
    Slider("n_doors", "Nombre de portes", 3, 15, 3),
    Slider("n_games", "Nombre de parties (mode Simuler)", 10, 100_000, 1000, log=True),
]
rng = np.random.default_rng()

# ------------------------------------------------------------------ le jeu
# La zone de dessin mesure 10 × 6 : un clic arrive à la fonction clic() dans ces coordonnées.
E = {}                                            # partie en cours
SCORE = {"garder": [0, 0], "changer": [0, 0]}     # [victoires, parties jouées]
BTN_G, BTN_C, BTN_R = (1.0, 0.45, 3.8, 0.8), (5.2, 0.45, 3.8, 0.8), (3.2, 0.45, 3.6, 0.8)   # x, y, largeur, hauteur
Y0, HP = 2.3, 2.5                                 # bas et hauteur des portes


def reset():
    E.clear()


def nouvelle(n):
    E.clear()
    E.update(n=n, prix=int(rng.integers(n)), choix=None, reste=None, phase=0, final=None, win=False)


def portes(n):
    pas = 9.0 / n
    return [(0.5 + (i + 0.5) * pas, pas * 0.72) for i in range(n)]      # (centre, largeur)


def dans(b, x, y):
    return b[0] <= x <= b[0] + b[2] and b[1] <= y <= b[1] + b[3]


def clic(x, y, mode, n_doors, **_):
    if mode != "Jouer":
        return
    if E.get("n") != n_doors:
        nouvelle(n_doors)
    if E["phase"] == 0:
        for i, (c, w) in enumerate(portes(n_doors)):
            if abs(x - c) <= w / 2 and Y0 <= y <= Y0 + HP:
                E["choix"] = i
                # porte laissée fermée : celle du prix, ou une chèvre au hasard si on a déjà le prix
                E["reste"] = E["prix"] if i != E["prix"] else (E["prix"] + int(rng.integers(1, n_doors))) % n_doors
                E["phase"] = 1
    elif E["phase"] == 1:
        for strat, bouton in (("garder", BTN_G), ("changer", BTN_C)):
            if dans(bouton, x, y):
                E["final"] = E["choix"] if strat == "garder" else E["reste"]
                E["win"] = E["final"] == E["prix"]
                SCORE[strat][1] += 1
                SCORE[strat][0] += int(E["win"])
                E["phase"] = 2
    elif dans(BTN_R, x, y):
        nouvelle(n_doors)


def jouer(fig, n):
    if E.get("n") != n:
        nouvelle(n)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ph, ch, re, pr = E["phase"], E["choix"], E["reste"], E["prix"]
    msg = ["Un prix est caché derrière une des portes. Clique sur la porte de ton choix.",
           f"Tu as choisi la porte {(ch or 0) + 1}. L’animateur ouvre les autres portes, sauf la porte {(re or 0) + 1}, "
           "et ne montre que des chèvres.",
           ("Gagné !" if E["win"] else "Perdu.") + f"  Le prix était derrière la porte {pr + 1}."][ph]
    ax.text(5, 5.5, msg, ha="center", va="center", fontsize=12)
    fs = max(6, min(11, 70 / n))
    for i, (c, w) in enumerate(portes(n)):
        ouverte = ph == 2 or (ph == 1 and i not in (ch, re))
        if ouverte:
            face = to_rgba("#f5a524", 0.9) if i == pr else to_rgba("#8b93a1", 0.35)
        else:
            face = to_rgba("#a0703c")
        bord, lw = "0.5", 1
        if ph >= 1 and i == ch:
            bord, lw = "#2563eb", 4
        if ph == 2 and i == E["final"]:
            bord, lw = ("#30a46c" if E["win"] else "#e5484d"), 4
        ax.add_patch(Rectangle((c - w / 2, Y0), w, HP, facecolor=face, edgecolor=bord, lw=lw))
        if ouverte:
            ax.text(c, Y0 + HP / 2, "★" if i == pr else "chèvre", ha="center", va="center",
                    fontsize=26 if i == pr else fs, color="white" if i == pr else None)
        ax.text(c, Y0 - 0.3, str(i + 1) + (" (ta porte)" if ph >= 1 and i == ch and n <= 8 else ""),
                ha="center", va="center", fontsize=fs + 1)

    def bouton(b, texte, couleur):
        ax.add_patch(Rectangle(b[:2], b[2], b[3], facecolor=couleur, edgecolor="none"))
        ax.text(b[0] + b[2] / 2, b[1] + b[3] / 2, texte, ha="center", va="center",
                fontsize=12, color="white", weight="bold")

    if ph == 1:
        bouton(BTN_G, f"Garder la porte {ch + 1}", "#2563eb")
        bouton(BTN_C, f"Changer pour la porte {re + 1}", "#e5484d")
    elif ph == 2:
        bouton(BTN_R, "Rejouer", "#2563eb")
    g, s = SCORE["garder"], SCORE["changer"]
    if g[1] + s[1]:
        ax.text(5, 0.12, f"Garder : {g[0]} gagnée(s) sur {g[1]}      Changer : {s[0]} gagnée(s) sur {s[1]}",
                ha="center", va="center", fontsize=10)


# ------------------------------------------------------------- la simulation
def simuler(fig, n_doors, n_games):
    prix = rng.integers(0, n_doors, n_games)            # porte du prix
    choix = rng.integers(0, n_doors, n_games)           # première porte choisie
    # Porte laissée fermée par l'animateur : celle du prix si le joueur s'est trompé,
    # sinon une porte à chèvre prise au hasard (décalage aléatoire de 1 à n-1 à partir du prix).
    reste = np.where(choix == prix, (prix + rng.integers(1, n_doors, n_games)) % n_doors, prix)
    garde = choix == prix
    change = reste == prix

    th = np.array([1 / n_doors, (n_doors - 1) / n_doors]) * 100
    freq = np.array([garde.mean(), change.mean()]) * 100

    ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.6]})
    ax1.bar([0, 1], freq, width=0.6, color=["C0", "C1"], alpha=0.75, label="fréquence observée")
    ax1.plot([0, 1], th, "D", color="C3", label="probabilité théorique")
    for x, f in enumerate(freq):
        ax1.text(x, f + 2, f"{f:.1f}%", ha="center", fontsize=10)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["garder", "changer"])
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Parties gagnées (%)")
    ax1.legend(loc="upper left", fontsize=8)

    n = np.arange(1, n_games + 1)
    ax2.plot(n, np.cumsum(garde) / n * 100, color="C0", label="garder")
    ax2.plot(n, np.cumsum(change) / n * 100, color="C1", label="changer")
    ax2.axhline(th[0], color="C0", ls="--", lw=1)
    ax2.axhline(th[1], color="C1", ls="--", lw=1)
    ax2.set_xscale("log")
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("Nombre de parties jouées")
    ax2.set_ylabel("Fréquence cumulée de victoires (%)")
    ax2.legend(loc="center right")
    fig.suptitle(f"{n_games:,} parties avec {n_doors} portes".replace(",", " "))


def draw(fig, mode, n_doors, n_games):
    if mode == "Simuler":
        simuler(fig, n_doors, n_games)
    else:
        jouer(fig, n_doors)
