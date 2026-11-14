"""Onde de matière 2D : paquet d'onde d'un électron, fentes, écran, impacts.

Résout  i dψ/dt = -(1/2) ∇²ψ + V ψ  (ħ = m = 1) par split-step FFT, avec bords
absorbants. Uniquement NumPy, pour tourner dans Pyodide.
Doit rester cohérent avec la constante N de onde.html.
"""
import numpy as np

# ---------------------------------------------------------------- paramètres
N = 192            # points de grille par direction (doit valoir N dans propagation_particule_quantique.html)
L = 60.0           # le domaine est [-L/2, L/2]²
DT = 0.05          # pas de temps
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


def get_lut():
    """Table de couleurs 256 × RGBA (type magma)."""
    t = np.linspace(0, 1, 256)
    ancres = [0, .25, .5, .75, 1]
    cols = np.array([[0, 0, 4], [80, 18, 123], [182, 54, 121],
                     [251, 136, 97], [252, 253, 191]], float)
    lut = np.empty((256, 4), np.uint8)
    for c in range(3):
        lut[:, c] = np.interp(t, ancres, cols[:, c])
    lut[:, 3] = 255
    return lut


LUT = get_lut()


# ------------------------------------------------------------- configuration
def setup(mode, a, d, k0, sy):
    """mode : 'free' | 'one' | 'two'. a : largeur de fente, d : écart centre à
    centre, k0 : impulsion, sy : largeur transverse du paquet."""
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
        T=1.3 * (XS - X0 + 4 * SIGMA_X) / k0,       # durée totale
        half=np.exp((-1j * V - W) * DT / 2),
        wall=(V > 0).T[::-1],                       # masque d'affichage du mur
        profil=np.zeros(N),                         # flux cumulé sur l'écran
        imp=np.empty((0, 2)),                       # impacts (y, z)
        vmax0=0.5 * float(np.max(np.abs(psi) ** 2)),
        norm=1.0,
    )
    S["top"] = S["vmax0"]


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
    for col in (IX_S,):
        m = tirets & ~wall[:, col]
        rgba[m, col] = (0, 220, 255, 255)
    if len(S["imp"]):                               # impacts allumés sur l'écran
        iy = np.clip(np.rint(S["imp"][:, 0] / dx).astype(int) + N // 2, 0, N - 1)
        rows = N - 1 - iy
        rgba[rows, IX_S] = (170, 255, 255, 255)
        rgba[rows, IX_S + 1] = (170, 255, 255, 255)
    return np.ascontiguousarray(rgba)


def frame(steps, auto):
    """Avance de `steps` pas (si la simulation n'est pas finie) et renvoie
    l'image RGBA N × N × 4."""
    for _ in range(int(steps)):
        if S["t"] >= S["T"]:
            break
        _pas()
    return _rendu(auto)


def stats():
    """[t, T, probabilité dans la boîte, échelle de couleur, nb d'impacts, fini]"""
    return [S["t"], S["T"], S["norm"], S["top"], len(S["imp"]),
            1.0 if S["t"] >= S["T"] else 0.0]


# --------------------------------------------------------- écran et impacts
def geometry():
    """[y de la première cellule, largeur de cellule, nombre de cellules]"""
    ys = x[YSEL]
    return [float(ys[0]), float(dx), float(len(ys))]


def screen_profile():
    """Densité de probabilité d'impact prédite (flux intégré, normalisée)."""
    P = S["profil"][YSEL].copy()
    tot = P.sum()
    return P / (tot * dx) if tot > 0 else P


def fire(n):
    """Tire n impacts indépendants selon la loi d'impact ; renvoie leur nombre total."""
    P = S["profil"][YSEL].copy()
    tot = P.sum()
    if tot <= 0 or n <= 0:
        return len(S["imp"])
    P /= tot
    ys = x[YSEL]
    idx = rng.choice(len(P), size=int(n), p=P)
    y = ys[idx] + (rng.random(int(n)) - 0.5) * dx
    z = rng.random(int(n))                          # fentes invariantes selon z
    S["imp"] = np.vstack([S["imp"], np.column_stack([y, z])])
    return len(S["imp"])


def impacts():
    return np.ascontiguousarray(S["imp"], dtype=np.float64)


def clear_impacts():
    S["imp"] = np.empty((0, 2))
