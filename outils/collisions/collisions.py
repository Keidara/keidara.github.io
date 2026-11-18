"""Collisions entre deux disques : choc de face (1D) ou décalé (2D), avec coefficient de restitution."""
import numpy as np
from matplotlib.patches import Circle
from kit import *

PARAMS = [
    Select("dim", "Collision", ["1D", "2D"]),
    Slider("m1", "Masse du disque 1 (bleu)", 0.5, 5.0, 1.0, step=0.1, unit="kg"),
    Slider("m2", "Masse du disque 2 (rouge)", 0.5, 5.0, 2.0, step=0.1, unit="kg"),
    Slider("v1", "Vitesse initiale du disque 1", -3.0, 3.0, 2.0, step=0.1, unit="m/s"),
    Slider("v2", "Vitesse initiale du disque 2", -3.0, 3.0, -1.0, step=0.1, unit="m/s"),
    Slider("b", "Décalage latéral b / (r₁ + r₂)", -1.0, 1.0, 0.5, step=0.05, actif=("dim", ["2D"])),
    Slider("e", "Coefficient de restitution e", 0.0, 1.0, 1.0, step=0.05),
    Slider("t", "Avancement", 0.0, 1.0, 0.0, step=0.005, anim=8),
]

T0 = 2.4          # durée entre le début et le contact (s) ; la scène dure 2·T0


def calcul(dim, m1, m2, v1, v2, b, e):
    """Positions initiales, instant et position du contact, vitesses avant (V) et après (W) le choc.
    Les vitesses initiales sont dirigées selon x ; le décalage b agit sur la direction du choc."""
    r1, r2 = 0.3 * np.sqrt(m1), 0.3 * np.sqrt(m2)
    R = r1 + r2
    dy = b * R if dim == "2D" else 0.0
    V1, V2 = np.array([v1, 0.0]), np.array([v2, 0.0])
    c = dict(r1=r1, r2=r2, V1=V1, V2=V2, W1=V1, W2=V2, choc=False, tc=np.inf, duree=2 * T0,
             p1=np.array([-4.0, -dy / 2]), p2=np.array([4.0, dy / 2]))
    if v1 - v2 > 1e-9 and abs(dy) < R:
        h = np.sqrt(R * R - dy * dy)
        c1, c2 = np.array([-h / 2, -dy / 2]), np.array([h / 2, dy / 2])     # centres au moment du contact
        n = (c2 - c1) / R                                                    # normale du disque 1 vers le disque 2
        vn = float((V1 - V2) @ n)                                            # vitesse relative d'approche (selon n)
        J = (1 + e) * vn / (1 / m1 + 1 / m2)                                 # impulsion échangée
        c.update(choc=True, tc=T0, c1=c1, c2=c2, p1=c1 - V1 * T0, p2=c2 - V2 * T0,
                 W1=V1 - J / m1 * n, W2=V2 + J / m2 * n)
    return c


def positions(c, tau):
    if c["choc"] and tau > c["tc"]:
        return c["c1"] + c["W1"] * (tau - c["tc"]), c["c2"] + c["W2"] * (tau - c["tc"])
    return c["p1"] + c["V1"] * tau, c["p2"] + c["V2"] * tau


def fig_collision(fig, dim, m1, m2, v1, v2, b, e, t):
    c = calcul(dim, m1, m2, v1, v2, b, e)
    tau = t * c["duree"]
    ax = fig.subplots()
    taus = np.linspace(0, c["duree"], 200)
    tr = [positions(c, s) for s in taus]
    for k, col in enumerate(("C0", "C1")):
        ax.plot([p[k][0] for p in tr], [p[k][1] for p in tr], ":", color=col, lw=1)
    q1, q2 = positions(c, tau)
    apres = c["choc"] and tau > c["tc"]
    for q, r, col, V in ((q1, c["r1"], "C0", c["W1"] if apres else c["V1"]),
                         (q2, c["r2"], "C1", c["W2"] if apres else c["V2"])):
        ax.add_patch(Circle(q, r, facecolor=col, alpha=0.55, edgecolor=col, lw=2))
        if np.hypot(V[0], V[1]) > 1e-6:
            ax.annotate("", xy=q + 0.6 * V, xytext=q, arrowprops=dict(arrowstyle="->", color=col, lw=2))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if not c["choc"]:
        ax.set_title("Pas de collision : les disques ne se rencontrent pas")
    else:
        ax.set_title("Après la collision" if apres else "Avant la collision")


def fig_bilan(fig, dim, m1, m2, v1, v2, b, e):
    c = calcul(dim, m1, m2, v1, v2, b, e)
    pa, pb = m1 * c["V1"] + m2 * c["V2"], m1 * c["W1"] + m2 * c["W2"]
    ea = 0.5 * (m1 * float(c["V1"] @ c["V1"]) + m2 * float(c["V2"] @ c["V2"]))
    eb = 0.5 * (m1 * float(c["W1"] @ c["W1"]) + m2 * float(c["W2"] @ c["W2"]))
    ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [2, 1]})
    x, w = np.arange(2), 0.35
    ax1.bar(x - w / 2, pa, w, color="C0", alpha=0.75, label="avant")
    ax1.bar(x + w / 2, pb, w, color="C1", alpha=0.75, label="après")
    ax1.axhline(0, color="0.5", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["px", "py"])
    ax1.set_ylabel("kg·m/s")
    ax1.set_title("Quantité de mouvement totale")
    ax1.legend(fontsize=9)
    ax2.bar([0, 1], [ea, eb], color=["C0", "C1"], alpha=0.75)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["avant", "après"])
    ax2.set_ylabel("J")
    perte = 1 - eb / ea if ea > 1e-12 else 0.0
    ax2.set_title(f"Énergie cinétique (perte : {perte:.0%})")
