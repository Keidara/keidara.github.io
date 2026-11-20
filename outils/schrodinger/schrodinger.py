"""Onde de matière 2D : paquet d'onde d'un électron, fentes, écran, impacts.

Résout  i dψ/dt = -(1/2) ∇²ψ + V ψ  (ħ = m = 1) par split-step FFT, avec bords
absorbants. Uniquement NumPy, pour tourner dans Pyodide.
"""
import numpy as np
from kit import *

PARAMS = [
    Select("mode", "Configuration", ["Propagation libre", "Une fente", "Deux fentes"], "Propagation libre"),
    Slider("a", "Largeur de fente", 0.8, 3.5, 1.5, step=0.1, actif=("mode", ["Une fente", "Deux fentes"])),
    Slider("d", "Écart entre fentes", 4, 12, 6, step=0.5, actif=("mode", ["Deux fentes"])),
    Slider("k0", "Impulsion k₀", 1.5, 4, 3, step=0.1),
    Slider("sy", "Largeur du paquet σy", 2, 8, 6, step=0.5),
    Slider("vitesse", "Vitesse", 1, 20, 4, unit="pas/image"),
    Check("auto", "Échelle de couleur automatique", False),
]

# ---------------------------------------------------------------- paramètres
N = 192            # points de grille par direction
L = 60.0           # le domaine est [-L/2, L/2]²
DT = 0.025         # pas de temps (garder V0·DT ≲ 1 : sinon le split-step déforme le mur)
T_tot = 25.0       # Durée totale
X0 = -18.0         # position initiale du paquet
SIGMA_X = 1.5      # largeur longitudinale du paquet
XW = -4.0          # position du mur
WALL_T = 1.0       # épaisseur du mur
V0 = 40.0          # hauteur du mur
XS = 18.0          # position de l'écran
W_BORD = 6.0       # épaisseur de la zone absorbante
W0 = 5.0           # intensité de l'absorption
YM = 24.0          # demi-hauteur de la zone de détection

# ---------------------------------------------------------------- pré-calculs
dx = L / N
x = (np.arange(N) - N // 2) * dx
X, Y = np.meshgrid(x, x, indexing="ij")            # tableaux indexés [ix, iy]
_k = 2 * np.pi * np.fft.fftfreq(N, dx)
KIN = np.exp(-0.5j * (_k[:, None] ** 2 + _k[None, :] ** 2) * DT)
_s = np.clip(np.abs(x) - (L / 2 - W_BORD), 0, None) / W_BORD
_w = W0 * _s ** 2
W = _w[:, None] + _w[None, :]                      # potentiel optique (absorption)
IX_S = int(np.argmin(np.abs(x - XS)))              # colonne de l'écran
YSEL = np.abs(x) <= YM                             # cellules de détection
rng = np.random.default_rng()

S = {}                                             # état de la simulation
PEND = {"n": 0, "lot": 1}                          # impacts restant à tirer, et taille des lots
MODES = {"Propagation libre": "free", "Une fente": "one", "Deux fentes": "two"}


def barre_onde():
    """Table de couleurs 1 × 256 × RGBA (type magma)."""
    t = np.linspace(0, 1, 256)
    ancres = [0, .25, .5, .75, 1]
    cols = np.array([[0, 0, 4], [80, 18, 123], [182, 54, 121],
                     [251, 136, 97], [252, 253, 191]], float)
    lut = np.empty((256, 4), np.uint8)
    for c in range(3):
        lut[:, c] = np.interp(t, ancres, cols[:, c])
    lut[:, 3] = 255
    return lut[None]


LUT = barre_onde()[0]


# ------------------------------------------------------------- configuration
def init_onde(mode, a, d, k0, sy):
    """mode : 'free' | 'one' | 'two'. a : largeur de fente, d : écart centre à
    centre, k0 : impulsion, sy : largeur transverse du paquet."""
    mode = MODES[mode]
    V = np.zeros((N, N))
    if mode != "free":
        mur = np.abs(X - XW) <= WALL_T / 2
        if mode == "one":
            ouvert = np.abs(Y) < a / 2
        else:
            ouvert = (np.abs(Y - d / 2) < a / 2) | (np.abs(Y + d / 2) < a / 2)
        V[mur & ~ouvert] = V0

    psi = np.exp(-(X - X0) ** 2 / (4 * SIGMA_X ** 2)
                 - Y ** 2 / (4 * sy ** 2) + 1j * k0 * X)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx * dx)

    S.clear()
    S.update(
        V=V, psi=psi, t=0.0,
        T=T_tot,                                    # durée totale
        half=np.exp((-1j * V - W) * DT / 2),
        wall=(V > 0).T[::-1],                       # masque d'affichage du mur
        profil=np.zeros(N),                         # flux cumulé sur l'écran
        imp=np.empty((0, 2)),                       # impacts (y, z)
        vmax0=0.5 * float(np.max(np.abs(psi) ** 2)),
        norm=1.0,
    )
    S["top"] = S["vmax0"]
    PEND["n"] = 0
    return ["ecran"]


# ------------------------------------------------------------------- calcul
def _pas():
    psi = S["psi"]
    psi = psi * S["half"]
    psi = np.fft.ifft2(np.fft.fft2(psi) * KIN)
    psi *= S["half"]
    S["psi"] = psi
    S["t"] += DT
    # flux de probabilité j_x à travers l'écran
    i = IX_S
    dpsi = (psi[i + 1] - psi[i - 1]) / (2 * dx)
    j = np.imag(np.conj(psi[i]) * dpsi)
    S["profil"] += np.clip(j, 0, None) * DT


def _rendu(auto):
    psi = S["psi"]
    I = psi.real ** 2 + psi.imag ** 2
    S["norm"] = float(I.sum() * dx * dx)
    top = max(float(I.max()) * 0.8, 1e-12) if auto else S["vmax0"]
    S["top"] = top
    idx = (np.sqrt(np.clip(I / top, 0, 1)) * 255).astype(np.uint8)
    rgba = LUT[idx.T[::-1]]                         # lignes = y (haut = y max)

    wall = S["wall"]
    rgba[wall] = (235, 235, 235, 255)               # mur
    tirets = (np.arange(N) // 3) % 2 == 0           # écran en pointillés
    m = tirets & ~wall[:, IX_S]
    rgba[m, IX_S] = (0, 220, 255, 255)
    return np.ascontiguousarray(rgba)


def frame_onde(lecture, vitesse, auto):
    """Avance de `vitesse` pas (en lecture, ou pour atteindre l'écran si des impacts sont demandés),
    tire les impacts demandés une fois l'onde arrivée, et renvoie l'image."""
    fini = S["t"] >= S["T"]
    avance = (lecture or PEND["n"] > 0) and not fini
    if avance:
        for _ in range(int(vitesse)):
            if S["t"] >= S["T"]:
                break
            _pas()
    fini = S["t"] >= S["T"]
    tire = False
    if fini and PEND["n"] > 0:                      # tirage progressif des impacts
        m = min(PEND["lot"], PEND["n"])
        fire(m)
        PEND["n"] -= m
        tire = True
    img = _rendu(auto)
    texte = (f"t = {S['t']:.1f} / {S['T']:.1f}   Probabilité dans la boîte : {S['norm']:.2f}   "
             f"Impacts : {len(S['imp'])}" + ("   — simulation terminée" if fini else ""))
    return img, dict(taille=N, fini=bool(fini), texte=texte, figures=["ecran"] if ((avance and fini) or tire) else [])


# --------------------------------------------------------- écran et impacts
def screen_profile():
    """Densité de probabilité d'impact prédite (flux intégré, normalisée)."""
    P = S["profil"][YSEL].copy()
    tot = P.sum()
    return P / (tot * dx) if tot > 0 else P


def fire(n):
    """Tire n impacts indépendants selon la loi d'impact."""
    P = S["profil"][YSEL].copy()
    tot = P.sum()
    if tot <= 0 or n <= 0:
        return
    P /= tot
    ys = x[YSEL]
    idx = rng.choice(len(P), size=int(n), p=P)
    y = ys[idx] + (rng.random(int(n)) - 0.5) * dx
    z = rng.random(int(n))                          # fentes invariantes selon z
    S["imp"] = np.vstack([S["imp"], np.column_stack([y, z])])


def _tirer(n):
    """Demande n impacts : la simulation avance si besoin, puis les impacts sont tirés par lots."""
    PEND["n"] += n
    PEND["lot"] = max(1, -(-n // 60))


def tirer_500():
    _tirer(500)


def tirer_5000():
    _tirer(5000)


def effacer():
    S["imp"] = np.empty((0, 2))
    PEND["n"] = 0


def fig_ecran(fig):
    """Écran de détection : impacts (vue de face) et histogramme comparé à la prédiction."""
    ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    ys = x[YSEL]
    y0, y1 = ys[0] - dx / 2, ys[-1] + dx / 2
    imp = S["imp"]
    ax1.set_facecolor("black")
    if len(imp):
        ax1.scatter(imp[:, 0], imp[:, 1], s=4, color="white", alpha=0.85, linewidths=0)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.set_title(f"Impacts sur l’écran (vue de face) : {len(imp)}")
    P = screen_profile()
    if len(imp):
        ax2.hist(imp[:, 0], bins=np.linspace(y0, y1, 61), density=True, alpha=0.55, color="C0", label="impacts")
    if P.any():
        ax2.plot(ys, P, color="C1", lw=2.5, label="prédiction (flux intégré)")
        ax2.legend(loc="upper right", fontsize=9)
    else:
        ax2.text(0.5, 0.5, "L’onde n’a pas encore atteint l’écran", transform=ax2.transAxes, ha="center")
    ax2.set_xlim(y0, y1)
    ax2.set_xlabel("y")
    ax2.set_ylabel("densité de probabilité")
