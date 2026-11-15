"""Diffraction par une fente et interférences par deux fentes (optique ondulatoire)."""
import numpy as np
from kit import *

TITLE = "Diffraction et fentes d’Young"
INTRO = ("Une onde plane de longueur d’onde λ traverse une fente de largeur a (ou deux fentes séparées de d) "
         "et forme sur un écran, à la distance D, une figure d’intensité. Pour une direction θ telle que "
         "sin θ = x / √(x² + D²) :\n\n"
         "une fente : I ∝ sinc²(a sin θ / λ) ;  deux fentes : I ∝ sinc²(a sin θ / λ) · cos²(π d sin θ / λ). "
         "Les franges sont modulées par l’enveloppe de diffraction de chaque fente.\n\n"
         "C’est la version optique de l’expérience de l’outil « Équation de Schrödinger 2D », où des électrons "
         "forment le même type de figure.")
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


def draw(fig, mode, lam, a, d, D, demi, env):
    lam_m, a_m, d_m = lam * 1e-9, a * 1e-6, d * 1e-6
    x = np.linspace(-demi, demi, 2000) * 1e-2            # position sur l'écran, en mètres
    sin_t = x / np.hypot(x, D)

    diffr = np.sinc(a_m * sin_t / lam_m) ** 2            # np.sinc(u) = sin(πu)/(πu)
    I = diffr * np.cos(np.pi * d_m * sin_t / lam_m) ** 2 if mode == "Deux fentes" else diffr

    ax, ax_f = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(x * 100, I, color="C0", label="intensité")
    if env and mode == "Deux fentes":
        ax.plot(x * 100, diffr, "--", color="C1", label="enveloppe de diffraction")
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
