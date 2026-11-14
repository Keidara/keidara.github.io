"""Solveur de l'équation de la chaleur 2D (différences finies explicites).
Chargé par chaleur.html via Pyodide."""
import numpy as np

N = 96
u = np.zeros((N + 2, N + 2), dtype=np.float32)   # grille avec une couche de cellules fantômes
steps = 0
last_top = 1.0

# Palette (noir -> jaune pâle) sous forme de table de 256 couleurs
_stops = np.array([[0,0,4],[40,11,84],[101,21,110],[159,42,99],
                   [212,72,66],[245,125,21],[250,193,39],[252,255,164]], dtype=float)
_t = np.linspace(0, len(_stops) - 1, 256)
LUT = np.stack([np.interp(_t, np.arange(len(_stops)), _stops[:, k]) for k in range(3)], axis=1).round().astype(np.uint8)

def get_lut():
    out = np.empty((1, 256, 4), dtype=np.uint8)
    out[0, :, :3] = LUT
    out[0, :, 3] = 255
    return out

def init_field(kind):
    global steps
    u[:] = 0
    steps = 0
    y, x = np.mgrid[0:N, 0:N] / (N - 1)
    g = lambda cx, cy, s: np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / s)
    if kind == "spot":
        f = g(.5, .5, .004)
    elif kind == "two":
        f = np.maximum(g(.3, .35, .003), .7 * g(.7, .65, .005))
    elif kind == "ring":
        d = np.hypot(x - .5, y - .5)
        f = np.exp(-((d - .25) ** 2) / .0015)
    elif kind == "square":
        f = ((abs(x - .5) < .12) & (abs(y - .5) < .12)).astype(float)
    elif kind == "left":
        f = (x < .06).astype(float)
    elif kind == "noise":
        f = np.random.rand(N, N) * (np.random.rand(N, N) < .5)
    else:
        f = np.zeros((N, N))
    u[1:-1, 1:-1] = f

def _ghosts(bc):
    if bc == "dirichlet":
        u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0
    elif bc == "neumann":
        u[0, :] = u[1, :];  u[-1, :] = u[-2, :]
        u[:, 0] = u[:, 1];  u[:, -1] = u[:, -2]
    else:  # périodique
        u[0, :] = u[-2, :]; u[-1, :] = u[1, :]
        u[:, 0] = u[:, -2]; u[:, -1] = u[:, 1]

def step(r, bc):
    """Un pas de temps : u <- u + r * laplacien(u), avec r = alpha*dt/dx^2 <= 0.25."""
    global steps
    _ghosts(bc)
    c = u[1:-1, 1:-1]
    u[1:-1, 1:-1] = c + r * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4 * c)
    steps += 1

def paint(cx, cy, rad):
    """Chauffe la plaque autour de (cx, cy), en coordonnées de grille."""
    j, i = np.mgrid[0:N, 0:N]
    w = np.exp(-((i + .5 - cx) ** 2 + (j + .5 - cy) ** 2) / (2 * (rad / 2) ** 2))
    u[1:-1, 1:-1] = np.maximum(u[1:-1, 1:-1], w.astype(np.float32))

def frame(r, n, bc, auto):
    """Avance de n pas puis renvoie l'image RGBA (N, N, 4)."""
    global last_top
    for _ in range(n):
        step(r, bc)
    inner = u[1:-1, 1:-1]
    last_top = max(float(inner.max()), 1e-6) if auto else 1.0
    idx = (np.clip(inner / last_top, 0, 1) * 255).round().astype(np.uint8)
    rgba = np.empty((N, N, 4), dtype=np.uint8)
    rgba[..., :3] = LUT[idx]
    rgba[..., 3] = 255
    return np.ascontiguousarray(rgba[::-1])   # y vers le haut

def stats():
    inner = u[1:-1, 1:-1]
    return [steps, float(inner.max()), float(inner.mean()), last_top]
