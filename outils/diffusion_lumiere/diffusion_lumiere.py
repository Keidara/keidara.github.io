"""Diffusion de la lumière : pourquoi le ciel est bleu, le Soleil couchant rouge et les nuages blancs."""
import numpy as np
from kit import *

H, C, K = 6.62607015e-34, 299792458.0, 1.380649e-23

PARAMS = [
    Slider("r", "Rayon des particules diffusantes", 1.0, 10000.0, 1.0, log=True, unit="nm"),
    Slider("m", "Atmosphère traversée par le Soleil (masses d’air)", 1.0, 40.0, 1.0, log=True),
]

LAM = np.arange(380.0, 781.0)
M_SRGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])


def planck(lam_nm, T):
    lam = np.asarray(lam_nm, float) * 1e-9
    return 2 * H * C ** 2 / lam ** 5 / np.expm1(H * C / (lam * K * T))


SOL = planck(LAM, 5772.0)
SOL = SOL / SOL.max()                                  # spectre du Soleil (corps noir à 5 772 K), normalisé


def _g(lam, mu, s1, s2):
    s = np.where(lam < mu, s1, s2)
    return np.exp(-0.5 * ((lam - mu) / s) ** 2)


def cmf(lam):
    """Fonctions colorimétriques CIE 1931 (approximation par gaussiennes de Wyman, Sloan et Shirley)."""
    x = 1.056 * _g(lam, 599.8, 37.9, 31.0) + 0.362 * _g(lam, 442.0, 16.0, 26.7) - 0.065 * _g(lam, 501.1, 20.4, 26.2)
    y = 0.821 * _g(lam, 568.8, 46.9, 40.5) + 0.286 * _g(lam, 530.9, 16.3, 31.1)
    z = 1.217 * _g(lam, 437.0, 11.8, 36.0) + 0.681 * _g(lam, 459.0, 26.0, 13.8)
    return x, y, z


def couleur(S):
    """Couleur sRGB d'un spectre (luminosité normalisée : on garde la teinte, pas l'éclat)."""
    xyz = np.array([np.sum(S * c) for c in cmf(LAM)])
    rgb = np.clip(M_SRGB @ xyz, 0, None)
    rgb = rgb / max(rgb.max(), 1e-12)
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return np.clip(rgb, 0, 1)


def lambda_vers_rgb(nm):
    if nm < 440:   r, g, b = (440 - nm) / 60, 0, 1
    elif nm < 490: r, g, b = 0, (nm - 440) / 50, 1
    elif nm < 510: r, g, b = 0, 1, (510 - nm) / 20
    elif nm < 580: r, g, b = (nm - 510) / 70, 1, 0
    elif nm < 645: r, g, b = 1, (645 - nm) / 65, 0
    else:          r, g, b = 1, 0, 0
    return np.array([r, g, b])


def efficacite(x):
    """Efficacité de diffusion Q en fonction du paramètre de taille x = 2πr/λ (modèle simplifié) :
    Q ≈ 0,11 x⁴ pour x ≪ 1 (Rayleigh), qui sature à 2 pour x ≫ 1 (grosses particules, diffusion blanche)."""
    q = 0.111 * np.asarray(x, float) ** 4
    return q / np.sqrt(1 + (q / 2) ** 2)


def tau(lam, m):
    """Épaisseur optique de l'air (diffusion de Rayleigh) : environ 0,1 à 550 nm à la verticale, en λ⁻⁴."""
    return 0.0975 * m * (550.0 / np.asarray(lam, float)) ** 4


def fig_spectres(fig, r, m):
    ax = fig.subplots()
    sca = SOL * efficacite(2 * np.pi * r / LAM)
    sca = sca / max(sca.max(), 1e-30)
    tra = SOL * np.exp(-tau(LAM, m))
    bande = np.array([[lambda_vers_rgb(n) for n in LAM]])
    ax.imshow(bande, extent=[380, 780, 0, 1.12], aspect="auto", alpha=0.35, zorder=0)
    ax.plot(LAM, SOL, color="0.4", lw=1.5, label="Soleil hors atmosphère")
    ax.plot(LAM, tra, color="C3", lw=2.5, label=f"Soleil après {m:.3g} masse{'s' if m >= 2 else ''} d’air")
    ax.plot(LAM, sca, "--", color="C0", lw=2.5, label="lumière diffusée (forme, unités arbitraires)")
    ax.set_xlim(380, 780)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("Longueur d’onde (nm)")
    ax.set_ylabel("Intensité relative")
    ax.set_title(f"Transmission : {100 * np.exp(-tau(450, m)):.0f} % à 450 nm (bleu), {100 * np.exp(-tau(650, m)):.0f} % à 650 nm (rouge)")
    ax.legend(loc="lower right", fontsize=9)


def fig_couleurs(fig, r, m):
    ax = fig.subplots()
    sca = SOL * efficacite(2 * np.pi * r / LAM)
    cols = [couleur(SOL), couleur(SOL * np.exp(-tau(LAM, m))), couleur(sca)]
    noms = ["Soleil hors atmosphère", f"Soleil après {m:.3g} masses d’air", f"Lumière diffusée (r = {r:.3g} nm)"]
    ax.imshow(np.array([cols]), aspect="auto", extent=[0, 3, 0, 1])
    for i, (c, nom) in enumerate(zip(cols, noms)):
        clair = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] > 0.5
        hexa = "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c)
        ax.text(i + 0.5, 0.5, f"{nom}\n{hexa}", ha="center", va="center", fontsize=11, color="black" if clair else "white")
    ax.axis("off")


def fig_efficacite(fig, r):
    ax = fig.subplots()
    x = np.logspace(-3, 2, 400)
    ax.loglog(x, efficacite(x), color="C0", lw=2, label="efficacité de diffusion Q")
    ax.loglog(x, 0.111 * x ** 4, ":", color="0.5", label="∝ x⁴ (Rayleigh, soit ∝ 1/λ⁴)")
    for lam, col, nom in ((450, "#3b6cf6", "bleu 450 nm"), (550, "#2fa84f", "vert 550 nm"), (650, "#e5484d", "rouge 650 nm")):
        xx = 2 * np.pi * r / lam
        ax.loglog([xx], [efficacite(xx)], "o", color=col, ms=9, label=nom)
    rb = efficacite(2 * np.pi * r / 450) / efficacite(2 * np.pi * r / 650)
    ax.set_ylim(1e-12, 5)
    ax.set_xlabel("Paramètre de taille x = 2πr / λ")
    ax.set_ylabel("Efficacité de diffusion Q")
    ax.set_title(f"Le bleu est diffusé {rb:.2f} fois plus que le rouge")
    ax.legend(loc="lower right", fontsize=9)
