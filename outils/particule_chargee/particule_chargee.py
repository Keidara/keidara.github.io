"""Particule chargée dans un champ magnétique B (selon z) et un champ électrique E, solution exacte.
Unités sans dimension : masse m = 1 et charge |q| = 1, donc la pulsation cyclotron vaut ω_c = B."""
import numpy as np
from kit import *

PARAMS = [
    Select("charge", "Charge de la particule", ["Positive", "Négative"]),
    Slider("B", "Champ magnétique B (selon z)", 0.0, 3.0, 1.0, step=0.05),
    Slider("Ex", "Champ électrique perpendiculaire E⊥ (selon x)", -2.0, 2.0, 0.4, step=0.05),
    Slider("Ez", "Champ électrique parallèle E∥ (selon z)", -1.0, 1.0, 0.0, step=0.05),
    Slider("vperp", "Vitesse initiale perpendiculaire à B", 0.0, 3.0, 1.0, step=0.1),
    Slider("vpar", "Vitesse initiale parallèle à B", -2.0, 2.0, 0.3, step=0.1),
    Slider("t", "Temps", 0.0, 30.0, 0.0, step=0.1, anim=30),
]

TMAX = 30.0


def trajectoire(tt, charge, B, Ex, Ez, vperp, vpar):
    """Position et vitesse à chaque instant tt. La particule part de l'origine avec la vitesse (vperp, 0, vpar).
    Dans le plan xy, le mouvement est une rotation à la pulsation ω = qB autour d'un point qui dérive à la vitesse E×B/B²."""
    q = 1.0 if charge == "Positive" else -1.0
    tt = np.asarray(tt, float)
    info = dict(w0=0.0, vd=0.0)
    if B > 1e-3:
        w = q * B
        vd = complex(0.0, -Ex / B)                       # vitesse de dérive E×B/B², indépendante de la charge
        w0 = complex(vperp, 0.0) - vd                    # vitesse initiale relative à la dérive
        e = np.exp(-1j * w * tt)
        v = vd + w0 * e
        z = vd * tt + w0 * (1 - e) / (1j * w)
        x, y, vx, vy = z.real, z.imag, v.real, v.imag
        info = dict(w0=abs(w0), vd=abs(vd))
    else:                                                # pas de champ magnétique : mouvement uniformément accéléré
        vx = vperp + q * Ex * tt
        vy = np.zeros_like(tt)
        x = vperp * tt + 0.5 * q * Ex * tt ** 2
        y = np.zeros_like(tt)
    vz = vpar + q * Ez * tt
    z3 = vpar * tt + 0.5 * q * Ez * tt ** 2
    return x, y, z3, vx, vy, vz, info


def fig_trajectoire(fig, charge, B, Ex, Ez, vperp, vpar, t):
    tt = np.linspace(0, TMAX, 3000)
    x, y, z, vx, vy, vz, info = trajectoire(tt, charge, B, Ex, Ez, vperp, vpar)
    ec = 0.5 * (vx ** 2 + vy ** 2 + vz ** 2)
    k = min(int(np.searchsorted(tt, t)), len(tt) - 1)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1])
    ax = fig.add_subplot(gs[:, 0])
    az = fig.add_subplot(gs[0, 1])
    ae = fig.add_subplot(gs[1, 1], sharex=az)

    ax.plot(x, y, color="C0", lw=1, alpha=0.3)
    ax.plot(x[:k + 1], y[:k + 1], color="C0", lw=2)
    ax.plot([0], [0], "s", color="0.4", ms=6)
    ax.plot([x[k]], [y[k]], "o", color="C3", ms=10)
    ax.set_aspect("equal", "datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if B > 1e-3:
        ax.set_title(f"Vue de dessus (B vers l’observateur)\nrayon de la spirale {info['w0'] / B:.2f}, dérive {info['vd']:.2f}", fontsize=12)
    else:
        ax.set_title("Vue de dessus : pas de champ magnétique", fontsize=12)

    az.plot(tt, z, color="C2", lw=1, alpha=0.3)
    az.plot(tt[:k + 1], z[:k + 1], color="C2", lw=2)
    az.plot([tt[k]], [z[k]], "o", color="C3", ms=8)
    az.set_ylabel("z")
    az.set_title("Mouvement selon B", fontsize=12)
    az.tick_params(labelbottom=False)

    ae.plot(tt, ec, color="C1", lw=1, alpha=0.3)
    ae.plot(tt[:k + 1], ec[:k + 1], color="C1", lw=2)
    ae.plot([tt[k]], [ec[k]], "o", color="C3", ms=8)
    ae.set_ylim(bottom=0)
    ae.set_xlabel("Temps")
    ae.set_ylabel("Énergie cinétique")
