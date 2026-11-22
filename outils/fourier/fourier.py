"""Fourier : décomposition d'un signal périodique en harmoniques, et filtrage d'une image dans le domaine des fréquences."""
import io
import numpy as np
from PIL import Image
from kit import *

SIGNAUX = ["Carré", "Triangle", "Dent de scie", "Impulsions"]
PARAMS = [
    Select("signal", "Signal", SIGNAUX),
    Slider("n_harm", "Nombre d’harmoniques", 1, 50, 5, anim=12),
    Select("filtre", "Filtre", ["Aucun", "Passe-bas", "Passe-haut", "Passe-bande"], "Passe-bas"),
    Slider("rayon", "Fréquence de coupure", 1, 100, 20, unit="%",
           actif=("filtre", ["Passe-bas", "Passe-haut", "Passe-bande"])),
    Slider("largeur", "Largeur de la bande", 1, 50, 8, unit="%", actif=("filtre", ["Passe-bande"])),
    Check("doux", "Bord adouci (limite les oscillations)", False),
]

# ------------------------------------------------------------------ signaux
M = 2048
_t = np.arange(M) / M


def forme(nom, t):
    """Un signal de période 1, d'amplitude entre −1 et 1 (sauf les impulsions : 0 ou 1)."""
    t = np.mod(t, 1.0)
    if nom == "Carré":
        return np.where(t < 0.5, 1.0, -1.0)
    if nom == "Triangle":
        return 1 - 4 * np.abs(t - 0.5)
    if nom == "Dent de scie":
        return 2 * t - 1
    return np.where(t < 0.25, 1.0, 0.0)


def coefficients(nom):
    """a0, a_n, b_n tels que f(t) = a0 + Σ a_n cos(2π n t) + b_n sin(2π n t), calculés par FFT."""
    c = np.fft.rfft(forme(nom, _t)) / M
    return c[0].real, 2 * c[1:51].real, -2 * c[1:51].imag


def reconstruction(nom, N, t):
    a0, a, b = coefficients(nom)
    n = np.arange(1, N + 1)[:, None]
    return a0 + (a[:N, None] * np.cos(2 * np.pi * n * t) + b[:N, None] * np.sin(2 * np.pi * n * t)).sum(axis=0)


def fig_serie(fig, signal, n_harm):
    ax = fig.subplots()
    t = np.linspace(0, 2, 1600)
    ax.plot(t, forme(signal, t), color="0.55", lw=1.5, label="Signal")
    ax.plot(t, reconstruction(signal, n_harm, t), color="C0", lw=2, label=f"Somme de {n_harm} harmonique{'s' if n_harm > 1 else ''}")
    ax.set_ylim(-1.5, 1.5) if signal != "Impulsions" else ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel(r"$t\,/\,T$")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Reconstruction du signal")


def fig_harmoniques(fig, signal, n_harm):
    ax = fig.subplots()
    a0, a, b = coefficients(signal)
    A, n = np.hypot(a, b), np.arange(1, 51)
    ax.bar(0, abs(a0), color="C2", label="Composante continue")
    ax.bar(n[:n_harm], A[:n_harm], color="C0", label="Harmoniques utilisées")
    ax.bar(n[n_harm:], A[n_harm:], color="0.6", label="Harmoniques ignorées")
    ax.set_xlabel(r"Rang $n$ de l’harmonique")
    ax.set_ylabel("Amplitude")
    ax.set_title("Spectre du signal")
    ax.legend(fontsize=9)


# -------------------------------------------------------------------- image
IMG = {}


def image_defaut():
    """Image d'essai : fond dégradé, disque, carré et zone rayée (qui donne des pics dans le spectre)."""
    y, x = np.mgrid[-1:1:192j, -1:1:192j]
    a = 0.35 + 0.15 * x
    a = np.where(np.hypot(x + 0.35, y + 0.3) < 0.28, 0.95, a)
    a = np.where((abs(x - 0.4) < 0.22) & (abs(y + 0.35) < 0.22), 0.1, a)
    rayures = (np.sin(2 * np.pi * 14 * (x + y)) > 0) * 0.35 + 0.1
    a = np.where((x < 0.15) & (y > 0.2), rayures, a)
    IMG["a"] = np.clip(a, 0, 1)


image_defaut()


def fichier_image(octets):
    """Image choisie par le visiteur : convertie en niveaux de gris et réduite à 256 pixels au plus."""
    im = Image.open(io.BytesIO(octets)).convert("L")
    im.thumbnail((256, 256))
    IMG["a"] = np.asarray(im, float) / 255
    return ["image"]


def fig_image(fig, filtre, rayon, largeur, doux):
    a = IMG["a"]
    h, w = a.shape
    F = np.fft.fftshift(np.fft.fft2(a))
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    r = np.hypot(fx, fy) / 0.5                       # 1 = fréquence de Nyquist (un pixel sur deux)
    rc, lg = rayon / 100, largeur / 100
    coupe = (lambda d: 1 / (1 + np.exp(np.clip(d / 0.02, -50, 50)))) if doux else (lambda d: (d <= 0).astype(float))
    masque = {"Aucun": np.ones_like(r), "Passe-bas": coupe(r - rc), "Passe-haut": 1 - coupe(r - rc),
              "Passe-bande": coupe(np.abs(r - rc) - lg)}[filtre]
    Ff = F * masque
    rec = np.clip(np.fft.ifft2(np.fft.ifftshift(Ff)).real, 0, 1)
    ax1, ax2, ax3 = fig.subplots(3, 1)
    ax1.imshow(a, cmap="gray", vmin=0, vmax=1)
    ax1.set_title("Image")
    ax2.imshow(np.log1p(np.abs(Ff)), cmap="magma", vmin=0, vmax=np.log1p(np.abs(F)).max())
    ax2.set_title("Spectre après filtre")
    ax3.imshow(rec, cmap="gray", vmin=0, vmax=1)
    ax3.set_title(f"Image filtrée ({100 * np.sum(np.abs(Ff) ** 2) / np.sum(np.abs(F) ** 2):.1f} % de l’énergie)")
    for ax in (ax1, ax2, ax3):
        ax.axis("off")
