"""Diffraction par une fente et interférences par deux fentes (optique ondulatoire)."""
import numpy as np
from kit import *

PARAMS = [
    Select("mode", "Configuration", ["Fente unique", "Deux fentes"], "Fente unique"),
    Slider("lam", r"Longueur d’onde \(\lambda\)", 400, 800, 550, unit="nm"),
    Slider("a", r"Largeur de fente \(a\)", 10, 300, 50, step=5, unit="µm"),
    Slider("d", r"Écart entre fentes \(d\)", 20, 1000, 250, step=10, unit="µm", actif=("mode", ["Deux fentes"])),    Slider("D", r"Distance à l’écran \(D\)", 0.5, 5.0, 2.0, step=0.1, unit="m"),
    Slider("largeur", r"Largeur de l’écran \(L\)", 1, 20, 6, unit="cm"),
    Check("env", "Afficher l’enveloppe de diffraction", True, actif=("mode", ["Deux fentes"]), cache=True),
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
    """Vue de dessus. Largeurs de fentes à l'échelle entre elles ; espacement des fronts d'onde
    proportionnel à λ ; distance fentes-écran proportionnelle à D (mais pas à l'échelle des fentes)."""
    from matplotlib.patches import Rectangle
    couleur = lambda_vers_rgb(lam)
    ech = 1 / 600                                        # 600 µm  <->  1 unité verticale
    x_mur = 3.0
    x_ecran = x_mur + 0.2 + 1.2 * D
    pas = 0.7 * lam / 550                                # distance entre deux fronts : proportionnelle à λ
    centres = [-d * ech / 2, d * ech / 2] if mode == "Deux fentes" else [0.0]
    bords = [-1.25]
    for c in centres:
        bords += [c - a * ech / 2, c + a * ech / 2]
    bords.append(1.25)
    for lo, hi in zip(bords[0::2], bords[1::2]):         # morceaux pleins du mur
        ax.add_patch(Rectangle((x_mur, lo), 0.18, hi - lo, color="0.45"))
    n = int((x_mur - 0.4 - 0.3) / pas) + 1               # au moins 3 fronts, davantage pour les petites λ
    xs = x_mur - 0.4 - pas * np.arange(n)
    for x in xs:
        ax.plot([x, x], [-0.8, 0.8], color=couleur, lw=2.5)
    ax.annotate("", xy=(xs[1], -0.95), xytext=(xs[0], -0.95), arrowprops=dict(arrowstyle="<->", lw=1))
    ax.text((xs[0] + xs[1]) / 2, -1.0, r"$\lambda$", ha="center", va="top", fontsize=11)
    ax.annotate("", xy=(2.8, 0), xytext=(0.3, 0), arrowprops=dict(arrowstyle="->", color=couleur, lw=1.5))
    for c in centres:                                    # rayons vers le centre de l'écran
        ax.plot([x_mur + 0.18, x_ecran], [c, 0], color=couleur, lw=1, alpha=0.6)
    ax.plot([x_ecran, x_ecran], [-1.25, 1.25], color="0.45", lw=4)
    ax.text(x_ecran + 0.1, 1.1, "écran", fontsize=10, va="top")
    ax.annotate("", xy=(x_ecran, -1.0), xytext=(x_mur + 0.1, -1.0), arrowprops=dict(arrowstyle="<->", lw=1))
    ax.text((x_mur + 0.1 + x_ecran) / 2, -0.95, f"$D$ = {D:.1f} m", ha="center", va="bottom", fontsize=10)
    titre = [f"$\\lambda$ = {lam} nm", f"$a$ = {a} µm"]
    if mode == "Deux fentes":
        titre.append(f"$d$ = {d} µm")
    ax.set_title(",   ".join(titre), fontsize=11)
    ax.set_xlim(0, 10.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axis("off")


def fig_schema(fig, mode, lam, a, d, D):
    schema(fig.subplots(), mode, lam, a, d, D)


def fig_intensite(fig, mode, lam, a, d, D, largeur, env):
    demi = largeur / 2                                   # l'écran est centré sur 0
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
    ax.set_ylabel(r"Intensité relative $I/I_0$")
    ax.legend(loc="upper right")
    titre = mode
    if mode == "Deux fentes" and a >= d:
        titre += r"  (attention : $a \geq d$, les fentes se chevauchent)"
    ax.set_title(titre)

    bande = I[None, :, None] * lambda_vers_rgb(lam)[None, None, :]    # image (1, N, 3)
    ax_f.imshow(np.clip(bande, 0, 1), extent=[-demi, demi, 0, 1], aspect="auto")
    ax_f.set_yticks([])
    ax_f.set_xlabel(r"Position $x$ sur l’écran (cm)")
    ax.set_xlim(-demi, demi)
    ax_f.set_xlim(-demi, demi)
    ax.tick_params(labelbottom=False)
