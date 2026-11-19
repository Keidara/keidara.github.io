"""Effet Doppler : source en mouvement émettant des ondes (vitesse de l'onde c = 1, période d'émission = 1)."""
import numpy as np
from matplotlib.patches import Circle
from kit import *

PARAMS = [
    Slider("M", "Vitesse de la source (en Mach : v / c)", 0.0, 2.5, 0.5, step=0.05),
    Slider("t", "Avancement", 0.0, 1.0, 0.0, step=0.005, anim=14),
]

XS0 = -6.0          # position de départ de la source
D_LAT = 6.0         # distance de l'observateur C à la trajectoire


def fig_ondes(fig, M, t):
    ax = fig.subplots()
    tau = t * 12 / max(M, 0.4)                       # la source traverse la scène
    for k in range(int(np.floor(tau)) + 1):          # un front d'onde émis à chaque période
        r = tau - k
        if r > 30:
            continue
        ax.add_patch(Circle((XS0 + M * k, 0), r, fill=False, edgecolor="C0", lw=1, alpha=0.8))
    xs = XS0 + M * tau
    if M > 1:                                         # cône de Mach : enveloppe des fronts d'onde
        a = np.arcsin(1 / M)
        for s in (1, -1):
            ax.plot([xs, xs - 30 * np.cos(a)], [0, s * 30 * np.sin(a)], "--", color="C1", lw=1.5)
    ax.plot([xs], [0], "o", color="C1", ms=11)
    for nom, (x, y) in (("A", (11, 0)), ("B", (-11, 0)), ("C", (0, D_LAT))):
        ax.plot([x], [y], "s", color="C2", ms=8)
        ax.annotate(nom, (x, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=11)
    ax.set_xlim(-12, 12)
    ax.set_ylim(-8, 8)
    ax.set_aspect("equal")
    ax.axis("off")
    if M < 1:
        ax.set_title(f"v = {M:.2f} c : longueur d’onde {1 - M:.2f} λ₀ devant la source, {1 + M:.2f} λ₀ derrière")
    elif M == 1:
        ax.set_title("v = c : les fronts d’onde s’accumulent devant la source (mur du son)")
    else:
        ax.set_title(f"v = {M:.2f} c : onde de choc, angle de Mach {np.degrees(np.arcsin(1 / M)):.1f}°")


def fig_frequence(fig, M):
    ax = fig.subplots()
    if M >= 1:
        ax.text(0.5, 0.5, "À vitesse supersonique, il n’y a plus de fréquence unique devant la source :\n"
                "l’observateur A reçoit d’un coup l’onde de choc, quand le cône de Mach passe.\n"
                "Derrière la source, l’observateur B entend toujours f₀ / (1 + M).",
                transform=ax.transAxes, ha="center", va="center", fontsize=11)
        ax.axis("off")
        return
    x = np.linspace(-12, 12, 500)
    ax.plot(x, 1 / (1 + M * x / np.hypot(x, D_LAT)), color="C0", lw=2, label="observateur C (la source passe devant lui)")
    ax.axhline(1 / (1 - M), ls="--", color="C1", label=f"observateur A (devant) : {1 / (1 - M):.2f} f₀")
    ax.axhline(1 / (1 + M), ls="--", color="C3", label=f"observateur B (derrière) : {1 / (1 + M):.2f} f₀")
    ax.axhline(1, ls=":", color="0.5")
    ax.set_xlabel("Position de la source à l’instant d’émission")
    ax.set_ylabel("Fréquence reçue / f₀")
    ax.set_title("Fréquence reçue par chaque observateur")
    ax.legend(loc="upper right", fontsize=9)
