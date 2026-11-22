"""Diffraction par une fente et interférences par deux fentes (optique ondulatoire)."""
import numpy as np
from kit import *

PARAMS = [
    Select("mode", "Configuration", ["Fente unique", "Deux fentes"], "Deux fentes"),
    Slider("lam", "Longueur d’onde", 380, 750, 550, unit="nm"),
    Slider("a", "Largeur de fente a", 10, 300, 50, step=5, unit="µm"),
    Slider("d", "Écart entre fentes d", 20, 1000, 250, step=10, unit="µm"),
    Slider("D", "Distance à l’écran D", 0.5, 5.0, 2.0, step=0.1, unit="m"),
    Slider("demi", "Demi-largeur d’écran vue", 0.5, 10.0, 3.0, step=0.5, unit="cm"),
    Check("env", "Afficher l’enveloppe de diffraction", True),
]


def lambda_vers_rgb(nm):
    """Couleur approchée d'une longueur d'onde visible (en nm), composantes entre 0 et 1."""
    if nm < 440:   r, g, b = (440 - nm) / 60, 0, 1
    elif nm < 490: r, g, b = 0, (nm - 440) / 50, 1
    elif nm < 510: r, g, b = 0, 1, (510 - nm) / 20
    elif nm < 580: r, g, b = (nm - 510) / 70, 1, 0
    elif nm < 645: r, g, b = 1, (645 - nm) / 65, 0
    else:          r, g, b = 1, 0, 0
    return np.array([r, g, b])


def schema(ax, mode, lam, a, d, D):
    """Vue de dessus de l'expérience. Largeurs de fentes à l'échelle entre elles, distances non."""
    from matplotlib.patches import Rectangle
    couleur = lambda_vers_rgb(lam)
    ech = 1 / 600                                        # 600 µm  <->  1 unité verticale
    centres = [-d * ech / 2, d * ech / 2] if mode == "Deux fentes" else [0.0]
    bords = [-1.25]
    for c in centres:
        bords += [c - a * ech / 2, c + a * ech / 2]
    bords.append(1.25)
    for lo, hi in zip(bords[0::2], bords[1::2]):         # morceaux pleins du mur
        ax.add_patch(Rectangle((3, lo), 0.18, hi - lo, color="0.45"))
    for x in (0.8, 1.5, 2.2):                            # fronts d'onde incidents
        ax.plot([x, x], [-0.9, 0.9], color=couleur, lw=2.5)
    ax.annotate("", xy=(2.7, 0), xytext=(0.3, 0), arrowprops=dict(arrowstyle="->", color=couleur, lw=1.5))
    for c in centres:                                    # rayons vers le centre de l'écran
        ax.plot([3.18, 9], [c, 0], color=couleur, lw=1, alpha=0.6)
    ax.plot([9, 9], [-1.25, 1.25], color="0.45", lw=4)
    ax.text(9.1, 1.1, "écran", fontsize=10, va="top")
    ax.annotate("", xy=(9, -1.0), xytext=(3.1, -1.0), arrowprops=dict(arrowstyle="<->", lw=1))
    ax.text(6, -0.9, f"D = {D:.1f} m (hors échelle)", ha="center", va="bottom", fontsize=10)
    ax.text(3.35, 1.2, f"a = {a} µm" + (f",  d = {d} µm" if mode == "Deux fentes" else ""), va="top", fontsize=10)
    ax.text(0.3, 1.2, f"λ = {lam} nm", va="top", fontsize=10)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.3, 1.3)
    ax.axis("off")


def fig_schema(fig, mode, lam, a, d, D):
    schema(fig.subplots(), mode, lam, a, d, D)


def fig_intensite(fig, mode, lam, a, d, D, demi, env):
    lam_m, a_m, d_m = lam * 1e-9, a * 1e-6, d * 1e-6
    x = np.linspace(-demi, demi, 2000) * 1e-2            # position sur l'écran, en mètres
    sin_t = x / np.hypot(x, D)

    diffr = np.sinc(a_m * sin_t / lam_m) ** 2            # np.sinc(u) = sin(πu)/(πu)
    I = diffr * np.cos(np.pi * d_m * sin_t / lam_m) ** 2 if mode == "Deux fentes" else diffr

    ax, ax_f = fig.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(x * 100, I, color="C0", label="Intensité")
    if env and mode == "Deux fentes":
        ax.plot(x * 100, diffr, "--", color="C1", label="Enveloppe de diffraction")
    ax.set_ylim(-0.03, 1.08)
    ax.set_ylabel("Intensité relative")
    ax.legend(loc="upper right")
    titre = mode
    if mode == "Deux fentes" and a >= d:
        titre += "  (attention : a ≥ d, les fentes se chevauchent)"
    ax.set_title(titre)

    bande = I[None, :, None] * lambda_vers_rgb(lam)[None, None, :]    # image (1, N, 3)
    ax_f.imshow(np.clip(bande, 0, 1), extent=[-demi, demi, 0, 1], aspect="auto")
    ax_f.set_yticks([])
    ax_f.set_xlabel("Position sur l’écran (cm)")
    ax.set_xlim(-demi, demi)
    ax_f.set_xlim(-demi, demi)
    ax.tick_params(labelbottom=False)
