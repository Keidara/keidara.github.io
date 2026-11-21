"""Rayonnement du corps noir : loi de Planck, loi de Wien et couleur perçue."""
import numpy as np
from kit import *

H, C, K = 6.62607015e-34, 299792458.0, 1.380649e-23    # Planck, vitesse de la lumière, Boltzmann
B_WIEN, SIGMA = 2.897771955e-3, 5.670374419e-8         # loi de Wien (m·K), Stefan-Boltzmann (W m⁻² K⁻⁴)

PARAMS = [
    Slider("T", "Température", 300, 20000, 5000, log=True, unit="K"),
    Check("log_x", "Échelle logarithmique en abscisse", False),
    Check("ref", "Comparer avec une ampoule (2 700 K) et le Soleil (5 772 K)", False),
]


def planck(lam_nm, T):
    """Luminance spectrale en kW / (m² · sr · nm)."""
    lam = np.asarray(lam_nm, float) * 1e-9
    with np.errstate(over="ignore"):
        B = 2 * H * C ** 2 / lam ** 5 / np.expm1(H * C / (lam * K * T))
    return B * 1e-12                                    # W m⁻² sr⁻¹ m⁻¹  ->  kW m⁻² sr⁻¹ nm⁻¹


def _g(lam, mu, s1, s2):
    s = np.where(lam < mu, s1, s2)
    return np.exp(-0.5 * ((lam - mu) / s) ** 2)


def cmf(lam):
    """Fonctions colorimétriques CIE 1931 (approximation par gaussiennes de Wyman, Sloan et Shirley, 2013)."""
    x = 1.056 * _g(lam, 599.8, 37.9, 31.0) + 0.362 * _g(lam, 442.0, 16.0, 26.7) - 0.065 * _g(lam, 501.1, 20.4, 26.2)
    y = 0.821 * _g(lam, 568.8, 46.9, 40.5) + 0.286 * _g(lam, 530.9, 16.3, 31.1)
    z = 1.217 * _g(lam, 437.0, 11.8, 36.0) + 0.681 * _g(lam, 459.0, 26.0, 13.8)
    return x, y, z


M_SRGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])


_LAM = np.arange(380.0, 781.0)
DECADES = 10          # nombre de décades d'éclat visible représentées sous la référence (3 000 K)


def _teinte_et_eclat(T):
    """Teinte sRGB (normalisée) et luminance visible Y d'un corps noir."""
    B = planck(_LAM, T)
    xyz = np.array([np.sum(B * c) for c in cmf(_LAM)])
    rgb = np.clip(M_SRGB @ xyz, 0, None)
    rgb = rgb / max(rgb.max(), 1e-300)
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return np.clip(rgb, 0, 1), max(float(xyz[1]), 1e-300)


Y_REF = _teinte_et_eclat(3000.0)[1]
_cache_couleur = {}


def couleur(T):
    """Couleur perçue d'un corps noir, éclat compris : la teinte est assombrie quand la lumière visible
    devient faible. L'éclat est tracé en échelle logarithmique (l'œil s'y adapte) et saturé au-dessus de 3 000 K,
    où la luminosité n'apporte plus d'information utile."""
    k = round(float(T), 1)
    if k not in _cache_couleur:
        rgb, Y = _teinte_et_eclat(T)
        niveau = np.clip(1 + np.log10(Y / Y_REF) / DECADES, 0, 1)
        _cache_couleur[k] = rgb * niveau
    return _cache_couleur[k]


def lambda_vers_rgb(nm):
    """Couleur approchée d'une longueur d'onde visible (en nm)."""
    if nm < 440:   r, g, b = (440 - nm) / 60, 0, 1
    elif nm < 490: r, g, b = 0, (nm - 440) / 50, 1
    elif nm < 510: r, g, b = 0, 1, (510 - nm) / 20
    elif nm < 580: r, g, b = (nm - 510) / 70, 1, 0
    elif nm < 645: r, g, b = 1, (645 - nm) / 65, 0
    else:          r, g, b = 1, 0, 0
    att = 0.3 + 0.7 * min(1, (nm - 380) / 40, (780 - nm) / 100)       # la vision s'éteint aux extrémités
    return np.array([r, g, b]) * att


def fig_spectre(fig, T, log_x, ref):
    ax = fig.subplots()
    lam_max = B_WIEN / T * 1e9                                      # loi de Wien, en nm
    if log_x:
        xmin, xmax = 50.0, 50000.0                                  # plage fixe : le pic reste dans la fenêtre de 300 K à 20 000 K
        lam = np.logspace(np.log10(xmin), np.log10(xmax), 2500)
    else:
        xmin, xmax = 0.0, float(np.clip(5 * lam_max, 1500, 30000))
        lam = np.linspace(50, xmax, 2500)
    courbes = [(T, f"T = {T:,.0f} K".replace(",", " "), "C0", 2.5)]
    if ref:
        courbes += [(2700, "Ampoule (2 700 K)", "C3", 1.2), (5772, "Soleil (5 772 K)", "C2", 1.2)]
    ymax = max(planck(lam, t).max() for t, *_ in courbes)
    bords = np.linspace(380, 780, 81)                               # bande du visible, en 80 tranches
    for a, b in zip(bords[:-1], bords[1:]):
        ax.axvspan(a, b, color=lambda_vers_rgb((a + b) / 2), alpha=0.45, lw=0, zorder=0)
    for t, lab, col, lw in courbes:
        ax.plot(lam, planck(lam, t), color=col, lw=lw, label=lab)
    ax.axvline(lam_max, ls=":", color="C1", lw=1.2, label=f"Pic (loi de Wien) : {lam_max:,.0f} nm".replace(",", " "))
    ax.text(580, ymax * 1.08, "visible", ha="center", va="top", fontsize=9)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, ymax * 1.12)
    ax.set_xlabel("Longueur d’onde (nm)")
    ax.set_ylabel("Luminance spectrale (kW · m⁻² · sr⁻¹ · nm⁻¹)")
    ax.set_title(f"{T:,.0f} K ({T - 273.15:,.0f} °C) : puissance rayonnée {SIGMA * T ** 4:.3g} W/m²".replace(",", " "))
    ax.legend(loc="upper right")


def fig_couleur(fig, T):
    ax1, ax2 = fig.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})
    rgb = couleur(T)
    ax1.imshow(rgb[None, None, :], aspect="auto")
    ax1.axis("off")
    clair = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] > 0.5
    ax1.text(0, 0, f"Couleur perçue d’un corps noir à {T:,.0f} K".replace(",", " "), ha="center", va="center",
             fontsize=14, color="black" if clair else "white")
    Ts = np.logspace(np.log10(300), np.log10(20000), 200)
    ax2.imshow(np.array([couleur(t) for t in Ts])[None], extent=[np.log10(300), np.log10(20000), 0, 1], aspect="auto")
    ax2.plot([np.log10(T)], [1.25], "v", color="C1", ms=10, clip_on=False)
    ax2.set_yticks([])
    ticks = [300, 500, 1000, 2000, 3000, 5000, 10000, 20000]
    ax2.set_xticks(np.log10(ticks))
    ax2.set_xticklabels([f"{t:,}".replace(",", " ") for t in ticks])
    ax2.set_xlabel("Température (K)")
