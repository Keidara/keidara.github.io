"""Gaz de billes 2D : disques durs en collisions élastiques dans une boîte. Loi de Maxwell, détente, mélange."""
import numpy as np
from collections import deque
from kit import *

PARAMS = [
    Select("depart", "Situation de départ", ["Vitesses égales", "Détente", "Chaud / froid", "Mélange"]),
    Slider("n_billes", "Nombre de billes", 20, 300, 150, step=10),
    Slider("v0", "Vitesse initiale typique", 0.1, 0.5, 0.3, step=0.05),
    Slider("vitesse", "Vitesse", 1, 20, 5, unit="pas/image"),
]

NPX = 320          # taille de l'image (pixels) ; la boîte est le carré [0, 1]²
R = 0.012          # rayon d'une bille
DT = 0.004         # pas de temps (une bille rapide fait moins d'un rayon par pas)
rng = np.random.default_rng()
S = {}
HIST = deque(maxlen=1500)                          # (temps, part restée à gauche, énergie gauche, énergie droite)

# disque de pixels dessiné pour chaque bille
_rp = R * NPX
_g = int(np.ceil(_rp))
_gy, _gx = np.mgrid[-_g:_g + 1, -_g:_g + 1]
_m = _gx ** 2 + _gy ** 2 <= _rp ** 2
OFF_X, OFF_Y = _gx[_m], _gy[_m]

# palette des vitesses : bleu (lent) -> rouge (rapide)
_cols = np.array([[70, 120, 255], [70, 220, 230], [250, 230, 90], [240, 70, 60]], float)
LUT = np.empty((256, 3), np.uint8)
for _c in range(3):
    LUT[:, _c] = np.interp(np.linspace(0, 1, 256), [0, .33, .66, 1.0], _cols[:, _c])
ORANGE, BLEU = np.array([245, 165, 36, 255], np.uint8), np.array([90, 160, 255, 255], np.uint8)


def _placer(n, x0, x1):
    """n positions au hasard dans [x0, x1] × [0, 1], sans chevauchement."""
    pos = np.column_stack([rng.uniform(x0 + R, x1 - R, n), rng.uniform(R, 1 - R, n)])
    lim = 2 * R * 1.02
    for _ in range(400):
        dx = pos[:, 0, None] - pos[None, :, 0]
        dy = pos[:, 1, None] - pos[None, :, 1]
        i, j = np.nonzero(np.triu(dx * dx + dy * dy < lim ** 2, 1))
        if len(i) == 0:
            break
        u = pos[i] - pos[j]
        nr = np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
        dep = (lim - nr) / 2 * (u / nr)
        np.add.at(pos, i, dep)
        np.add.at(pos, j, -dep)
        pos[:, 0] = np.clip(pos[:, 0], x0 + R, x1 - R)
        pos[:, 1] = np.clip(pos[:, 1], R, 1 - R)
    return pos


def init_gaz(depart, n_billes, v0):
    n = int(n_billes)
    if depart == "Détente":
        pos = _placer(n, 0.0, 0.5)
    elif depart == "Mélange":
        pos = np.vstack([_placer(n // 2, 0.0, 0.5), _placer(n - n // 2, 0.5, 1.0)])
    else:
        pos = _placer(n, 0.0, 1.0)
    vit = np.full(n, float(v0))
    if depart == "Chaud / froid":
        vit = np.where(pos[:, 0] < 0.5, 1.6 * v0, 0.5 * v0)
    ang = rng.uniform(0, 2 * np.pi, n)
    v = vit[:, None] * np.column_stack([np.cos(ang), np.sin(ang)])
    S.clear()
    S.update(x=pos.astype(float), v=v.astype(float), t=0.0, coll=0, n=n,
             gauche0=pos[:, 0] < 0.5, melange=(depart == "Mélange"),
             vref=float(np.sqrt((v ** 2).sum(1).mean())))
    HIST.clear()
    _mesure()
    return ["vitesses", "evolution"]


def _mesure():
    x, v = S["x"], S["v"]
    g = x[:, 0] < 0.5
    e = 0.5 * (v ** 2).sum(1)
    eg = float(e[g].mean()) if g.any() else np.nan
    ed = float(e[~g].mean()) if (~g).any() else np.nan
    HIST.append((S["t"], float(g[S["gauche0"]].mean()), eg, ed))


def _pas():
    x, v = S["x"], S["v"]
    x += v * DT
    for k in (0, 1):                                # murs : rebond élastique
        bas = x[:, k] < R
        x[bas, k] = 2 * R - x[bas, k]
        v[bas, k] = np.abs(v[bas, k])
        haut = x[:, k] > 1 - R
        x[haut, k] = 2 * (1 - R) - x[haut, k]
        v[haut, k] = -np.abs(v[haut, k])
    dx = x[:, 0, None] - x[None, :, 0]
    dy = x[:, 1, None] - x[None, :, 1]
    proche = np.triu(dx * dx + dy * dy < (2 * R) ** 2, 1)
    for a, b in zip(*np.nonzero(proche)):           # collisions : on échange les composantes normales
        u = x[a] - x[b]
        d = float(np.hypot(u[0], u[1]))
        if d < 1e-12:
            continue
        u = u / d
        vn = float((v[a] - v[b]) @ u)
        if vn < 0:
            v[a] -= vn * u
            v[b] += vn * u
            S["coll"] += 1
        rec = (2 * R - d) / 2
        x[a] += u * rec
        x[b] -= u * rec
    S["t"] += DT


def _rendu():
    img = np.empty((NPX, NPX, 4), np.uint8)
    img[:] = (12, 14, 20, 255)
    tirets = (np.arange(NPX) // 4) % 2 == 0         # moitié gauche / moitié droite
    img[tirets, NPX // 2] = (70, 76, 92, 255)
    x, v = S["x"], S["v"]
    ix = np.clip((x[:, 0] * NPX).astype(int), 0, NPX - 1)
    iy = np.clip(((1 - x[:, 1]) * NPX).astype(int), 0, NPX - 1)
    if S["melange"]:
        col = np.where(S["gauche0"][:, None], ORANGE[None, :], BLEU[None, :])
    else:
        sp = np.hypot(v[:, 0], v[:, 1])
        idx = (np.clip(sp / (2.5 * S["vref"]), 0, 1) * 255).astype(int)
        col = np.concatenate([LUT[idx], np.full((len(idx), 1), 255, np.uint8)], axis=1)
    rows = np.clip(iy[:, None] + OFF_Y[None, :], 0, NPX - 1)
    cols = np.clip(ix[:, None] + OFF_X[None, :], 0, NPX - 1)
    img[rows, cols] = col[:, None, :]
    return np.ascontiguousarray(img)


def frame_gaz(lecture, vitesse):
    if lecture:
        for _ in range(int(vitesse)):
            _pas()
        _mesure()
    vr = float(np.sqrt((S["v"] ** 2).sum(1).mean()))
    texte = (f"t = {S['t']:.1f}   Billes : {S['n']}   Vitesse quadratique moyenne : {vr:.3f}   "
             f"Collisions : {S['coll']:,}".replace(",", " "))
    return _rendu(), dict(taille=NPX, fini=False, texte=texte, figures=["vitesses", "evolution"] if lecture else [])


def fig_vitesses(fig):
    ax = fig.subplots()
    v = np.hypot(S["v"][:, 0], S["v"][:, 1])
    vr = float(np.sqrt(np.mean(v ** 2)))
    xmax = 3.2 * vr
    ax.hist(v, bins=np.linspace(0, xmax, 41), density=True, alpha=0.75, color="C0", label="vitesses des billes")
    vv = np.linspace(0, xmax, 300)
    ax.plot(vv, 2 * vv / vr ** 2 * np.exp(-vv ** 2 / vr ** 2), color="C1", lw=2, label="loi de Maxwell (2D)")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.8 * 0.858 / vr)
    ax.set_xlabel("Vitesse |v|")
    ax.set_ylabel("Densité de probabilité")
    ax.set_title(f"{S['n']} billes, vitesse quadratique moyenne {vr:.3f}")
    ax.legend(loc="upper right", fontsize=9)


def fig_evolution(fig):
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    h = np.array(HIST) if HIST else np.empty((0, 4))
    if len(h):
        ax1.plot(h[:, 0], h[:, 1], color="C0", lw=2)
        ax2.plot(h[:, 0], h[:, 2], color="C1", lw=2, label="moitié gauche")
        ax2.plot(h[:, 0], h[:, 3], color="C2", lw=2, label="moitié droite")
    em = 0.5 * float((S["v"] ** 2).sum(1).mean())
    ax1.axhline(0.5, ls=":", color="0.5")
    ax1.set_ylim(0.35, 1.05)
    ax1.set_ylabel("part restée à gauche")
    ax1.set_title("Billes parties de la moitié gauche, et énergie cinétique par moitié")
    ax1.tick_params(labelbottom=False)
    ax2.axhline(em, ls=":", color="0.5", label="moyenne générale")
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel("énergie cinétique moyenne")
    ax2.set_xlabel("Temps")
    ax2.legend(loc="upper right", fontsize=9)
