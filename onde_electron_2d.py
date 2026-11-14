#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paquet d'onde d'un électron en 2D : propagation libre, une fente, deux fentes.

Résout  i dψ/dt = -(1/2)∇²ψ + V ψ   (unités atomiques naturelles : ħ = m = 1)
par la méthode « split-step » (FFT), avec bords absorbants (potentiel optique
imaginaire) pour éviter les réflexions / le repliement périodique.

Un écran de détection est placé à droite : on y accumule le flux de probabilité
au cours du temps -> distribution de probabilité d'impact P(y).
On peut ensuite « tirer » N électrons un par un : chaque impact est un tirage
aléatoire selon P(y) -> on voit le motif d'interférences émerger point par point.

Dépendances : numpy, matplotlib (scipy recommandé pour la vitesse)
Usage       : python onde_electron_2d.py [--mode libre|1|2]
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.widgets import Button, RadioButtons, Slider

try:
    import scipy.fft as fft
    FFT_KW = {"workers": -1}
except ImportError:                       # repli sur numpy
    import numpy.fft as fft
    FFT_KW = {}

# ----------------------------------------------------------------- paramètres
N = 384            # points de grille par direction
L = 80.0           # taille du domaine [-L/2, L/2]²
DT = 0.04          # pas de temps
NSUB = 6           # pas de temps entre deux rafraîchissements d'image
X0 = -25.0         # position initiale du paquet (x)
SIGMA_X = 2.0      # largeur longitudinale du paquet
XW = -5.0          # position du mur
WALL_T = 1.0       # épaisseur du mur
V0 = 50.0          # hauteur du mur (≫ énergie cinétique)
XS = 25.0          # position de l'écran
W_BORD = 8.0       # épaisseur de la zone absorbante
W0 = 4.0           # intensité de l'absorption
YM = 32.0          # demi-hauteur affichée / détectée sur l'écran

MODES = ("Propagation libre", "Une fente", "Deux fentes")


# --------------------------------------------------------------------- solveur
class Solveur:
    def __init__(self):
        self.dx = L / N
        x = (np.arange(N) - N // 2) * self.dx
        self.x = x
        self.X, self.Y = np.meshgrid(x, x, indexing="ij")     # psi[ix, iy]
        k = 2 * np.pi * np.fft.fftfreq(N, self.dx)
        K2 = k[:, None] ** 2 + k[None, :] ** 2
        self.kin = np.exp(-0.5j * K2 * DT).astype(np.complex64)
        s = np.clip(np.abs(x) - (L / 2 - W_BORD), 0, None) / W_BORD
        w1 = W0 * s ** 2
        self.W = w1[:, None] + w1[None, :]                     # absorption
        self.ix_s = int(np.argmin(np.abs(x - XS)))

    def configurer(self, mode, a, d, k0, sy):
        """mode: 0 libre, 1 une fente, 2 deux fentes ; a: largeur fente ;
        d: distance centre à centre ; k0: impulsion ; sy: largeur transverse."""
        X, Y = self.X, self.Y
        V = np.zeros((N, N))
        if mode > 0:
            mur = np.abs(X - XW) <= WALL_T / 2
            if mode == 1:
                ouvert = np.abs(Y) < a / 2
            else:
                ouvert = (np.abs(Y - d / 2) < a / 2) | (np.abs(Y + d / 2) < a / 2)
            V[mur & ~ouvert] = V0
        self.V = V
        self.half = np.exp((-1j * V - self.W) * DT / 2).astype(np.complex64)

        psi = np.exp(-(X - X0) ** 2 / (4 * SIGMA_X ** 2)
                     - Y ** 2 / (4 * sy ** 2) + 1j * k0 * X)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * self.dx ** 2)
        self.psi = psi.astype(np.complex64)
        self.vmax = 0.5 * float(np.max(np.abs(self.psi) ** 2))
        self.t = 0.0
        self.T = 1.3 * (XS - X0 + 4 * SIGMA_X) / k0            # durée totale
        self.profil = np.zeros(N)                              # flux cumulé sur l'écran

    def pas(self):
        psi = self.psi
        psi *= self.half
        psi = fft.ifft2(fft.fft2(psi, **FFT_KW) * self.kin, **FFT_KW)
        psi *= self.half
        self.psi = psi
        self.t += DT
        # flux de probabilité j_x à travers l'écran
        i = self.ix_s
        dpsi = (psi[i + 1] - psi[i - 1]) / (2 * self.dx)
        j = np.imag(np.conj(psi[i]) * dpsi)
        self.profil += np.clip(j, 0, None) * DT

    @property
    def fini(self):
        return self.t >= self.T


# ----------------------------------------------------------------- interface
class App:
    def __init__(self, mode0=2):
        self.sim = Solveur()
        self.rng = np.random.default_rng()
        self.gen = 0                      # jeton pour interrompre une simulation
        self.mode = mode0
        self.imp_y = np.empty(0)
        self.imp_z = np.empty(0)
        self._build_figure()
        self.prepare()

    # -- construction de la figure --------------------------------------
    def _build_figure(self):
        fig = plt.figure(figsize=(13.5, 8.2))
        self.fig = fig
        fig.canvas.manager.set_window_title("Fonction d'onde d'un électron – 2D")
        self.ax = fig.add_axes([0.05, 0.31, 0.50, 0.64])
        self.ax_s = fig.add_axes([0.575, 0.31, 0.11, 0.64])
        self.ax_h = fig.add_axes([0.72, 0.31, 0.25, 0.64])

        ext = [-L / 2, L / 2, -L / 2, L / 2]
        z = np.zeros((N, N))
        self.im = self.ax.imshow(z.T, origin="lower", extent=ext, cmap="magma",
                                 aspect="auto", interpolation="bilinear",
                                 norm=PowerNorm(0.5, vmin=0, vmax=1))
        self.im_wall = self.ax.imshow(np.ma.masked_all((N, N)).T, origin="lower",
                                      extent=ext, cmap="gray", vmin=0, vmax=1,
                                      aspect="auto", interpolation="nearest")
        self.ax.axvline(XS, color="cyan", ls="--", lw=1)
        self.ax.text(XS + 0.8, YM - 3, "écran", color="cyan", fontsize=9)
        self.pts_main = self.ax.scatter([], [], s=6, c="cyan", zorder=5)
        self.ax.set_xlim(-L / 2, L / 2)
        self.ax.set_ylim(-YM, YM)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.titre = self.ax.set_title("")

        self.ax_s.set_facecolor("black")
        self.pts_s = self.ax_s.scatter([], [], s=5, c="white")
        self.ax_s.set_xlim(0, 1)
        self.ax_s.set_ylim(-YM, YM)
        self.ax_s.set_xticks([])
        self.ax_s.set_yticklabels([])
        self.ax_s.set_title("écran (vue de face)", fontsize=9)

        # contrôles
        ax_r = fig.add_axes([0.03, 0.03, 0.16, 0.18])
        self.radio = RadioButtons(ax_r, MODES, active=self.mode)
        self.radio.on_clicked(self._on_mode)

        def slider(y, label, lo, hi, v0, step=None, on=True):
            s = Slider(fig.add_axes([0.30, y, 0.25, 0.03]), label, lo, hi,
                       valinit=v0, valstep=step)
            if on:
                s.on_changed(lambda _v: self.prepare())
            return s

        self.s_a = slider(0.235, "largeur fente a", 0.5, 4.0, 1.5, 0.1)
        self.s_d = slider(0.185, "écart fentes d", 5.0, 16.0, 8.0, 0.5)
        self.s_k = slider(0.135, "impulsion k₀", 1.5, 5.0, 3.0, 0.1)
        self.s_sy = slider(0.085, "largeur paquet σy", 2.0, 10.0, 7.0, 0.5)
        self.s_n = slider(0.035, "nb d'impacts", 10, 5000, 500, 10, on=False)

        def bouton(x, y, txt, cb):
            b = Button(fig.add_axes([x, y, 0.16, 0.06]), txt)
            b.on_clicked(cb)
            return b

        self.b_sim = bouton(0.64, 0.19, "▶ Simuler l'onde", lambda _e: self.simuler())
        self.b_fire = bouton(0.82, 0.19, "⚛ Tirer N impacts", lambda _e: self.tirer())
        self.b_clear = bouton(0.64, 0.09, "Effacer les impacts", lambda _e: self.effacer())
        self.b_stop = bouton(0.82, 0.09, "■ Stop", lambda _e: self._stop())

    # -- état / dessin ----------------------------------------------------
    def _on_mode(self, label):
        self.mode = MODES.index(label)
        self.prepare()

    def _stop(self):
        self.gen += 1

    def _vivant(self):
        return plt.fignum_exists(self.fig.number)

    def prepare(self):
        """(Re)construit la simulation avec les paramètres courants."""
        self.gen += 1
        self.sim.configurer(self.mode, self.s_a.val, self.s_d.val,
                            self.s_k.val, self.s_sy.val)
        self.effacer(redraw=False)
        wall = np.ma.masked_where(self.sim.V <= 0, np.ones((N, N)))
        self.im_wall.set_data(wall.T)
        self.im.norm.vmax = self.sim.vmax
        self.afficher()
        self.fig.canvas.draw_idle()

    def afficher(self):
        s = self.sim
        self.im.set_data((np.abs(s.psi) ** 2).T)
        norme = float(np.sum(np.abs(s.psi) ** 2) * s.dx ** 2)
        self.titre.set_text(f"|ψ(x,y)|²   t = {s.t:5.1f} / {s.T:.1f}   "
                            f"probabilité restante = {norme:.2f}")
        self.dessiner_hist()

    def dessiner_hist(self):
        ax, s = self.ax_h, self.sim
        ax.cla()
        P = s.profil.copy()
        P[np.abs(s.x) > YM] = 0
        if P.sum() > 0:
            ax.plot(P / (P.sum() * s.dx), s.x, color="tab:red", lw=1.6,
                    label="prédiction (|ψ|² intégré)")
        if len(self.imp_y):
            ax.hist(self.imp_y, bins=np.linspace(-YM, YM, 65), density=True,
                    orientation="horizontal", color="tab:blue", alpha=0.6,
                    label="impacts")
        ax.set_ylim(-YM, YM)
        ax.set_yticklabels([])
        ax.set_xlabel("densité de probabilité")
        ax.set_title(f"{len(self.imp_y)} impact(s) sur l'écran", fontsize=10)
        if P.sum() > 0 or len(self.imp_y):
            ax.legend(loc="lower right", fontsize=8)

    def effacer(self, redraw=True):
        self.imp_y = np.empty(0)
        self.imp_z = np.empty(0)
        self.pts_main.set_offsets(np.empty((0, 2)))
        self.pts_s.set_offsets(np.empty((0, 2)))
        if redraw:
            self.dessiner_hist()
            self.fig.canvas.draw_idle()

    # -- simulation de l'onde --------------------------------------------
    def simuler(self):
        """Fait évoluer l'onde jusqu'à T. Renvoie False si interrompu."""
        if self.sim.fini:
            self.prepare()
        self.gen += 1
        mon = self.gen
        while not self.sim.fini:
            for _ in range(NSUB):
                self.sim.pas()
            self.afficher()
            if not self._vivant():
                return False
            plt.pause(0.001)
            if mon != self.gen:
                return False
        return True

    # -- impacts ------------------------------------------------------------
    def tirer(self):
        if not self.sim.fini:
            if not self.simuler():
                return
        s = self.sim
        P = s.profil.copy()
        P[np.abs(s.x) > YM] = 0
        if P.sum() <= 0:
            return
        P /= P.sum()
        self.gen += 1
        mon = self.gen
        n = int(self.s_n.val)
        lot = max(1, n // 60)
        fait = 0
        while fait < n:
            m = min(lot, n - fait)
            idx = self.rng.choice(N, size=m, p=P)
            y = s.x[idx] + (self.rng.random(m) - 0.5) * s.dx
            z = self.rng.random(m)             # fentes invariantes selon z
            self.imp_y = np.concatenate([self.imp_y, y])
            self.imp_z = np.concatenate([self.imp_z, z])
            self.pts_s.set_offsets(np.column_stack([self.imp_z, self.imp_y]))
            self.pts_main.set_offsets(
                np.column_stack([np.full(len(self.imp_y), XS), self.imp_y]))
            self.dessiner_hist()
            fait += m
            if not self._vivant():
                return
            plt.pause(0.01)
            if mon != self.gen:
                return


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["libre", "1", "2"], default="2",
                    help="configuration initiale (défaut : deux fentes)")
    args = ap.parse_args()
    App({"libre": 0, "1": 1, "2": 2}[args.mode])
    plt.show()


if __name__ == "__main__":
    main()
