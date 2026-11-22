"""Réfraction : lois de la réflexion et de Snell-Descartes, construction de Huygens-Fresnel."""
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Arc, Rectangle
from kit import *

PARAMS = [
    Slider("n1", r"Indice du milieu 1 (haut), \(n_1\)", 1.0, 2.5, 1.0, step=0.01),
    Slider("n2", r"Indice du milieu 2 (bas), \(n_2\)", 1.0, 2.5, 1.5, step=0.01),
    Slider("theta", r"Angle d’incidence \(\theta_1\)", 0, 89, 40, unit="°"),
    Check("refl", "Montrer aussi les ondelettes réfléchies", False),
    Slider("t", "Avancement", 0.0, 1.0, 0.0, step=0.005, anim=12),
]

X0, X1 = -9.0, 9.0          # étendue des sources secondaires sur l'interface


def angle_refracte(n1, n2, theta):
    """Angle θ₂ en degrés, ou None s'il y a réflexion totale."""
    s2 = n1 * np.sin(np.radians(theta)) / n2
    return None if s2 > 1 else float(np.degrees(np.arcsin(s2)))


def reflexion(n1, n2, th):
    """Facteurs de réflexion en intensité (polarisation s, polarisation p, moyenne), pour des angles th en radians.
    Sert ici à l'épaisseur des flèches ; le détail est dans l'outil sur la polarisation."""
    th = np.asarray(th, float)
    s2 = n1 * np.sin(th) / n2
    tir = s2 > 1
    c2 = np.sqrt(1 - np.minimum(s2, 1) ** 2)
    c1 = np.cos(th)
    rs = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)
    rp = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)
    Rs, Rp = np.where(tir, 1.0, rs ** 2), np.where(tir, 1.0, rp ** 2)
    return Rs, Rp, (Rs + Rp) / 2


def fig_huygens(fig, n1, n2, theta, refl, t):
    # Figure animée : on évite la mise en page automatique (coûteuse) et on regroupe les tracés en collections.
    try:
        fig.set_layout_engine("none")
    except AttributeError:                                  # matplotlib < 3.6
        fig.set_constrained_layout(False)
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.93])
    th = np.radians(theta)
    s1, c1 = np.sin(th), np.cos(th)
    v1, v2 = 1.0, n1 / n2                                   # vitesses de l'onde (la fréquence est la même : λ₁ = 1)
    tau = 12 * t
    th2 = angle_refracte(n1, n2, theta)
    ax.axhspan(-6, 0, color="C0", alpha=0.05 + 0.12 * (n2 - 1) / 1.5)
    ax.axhline(0, color="0.4", lw=2)

    # fronts d'onde incidents : un segment par front, découpé par la fenêtre (le front en gras sert à la construction)
    m = np.arange(10)
    s = X0 * s1 + v1 * tau - m
    u = np.array([-40.0, 40.0])
    segs = np.stack([s[:, None] * s1 + u[None, :] * c1, -s[:, None] * c1 + u[None, :] * s1], axis=-1)
    couleurs = [to_rgba("C0", 1.0 if k == 0 else 0.5) for k in m]
    fronts = LineCollection(segs, colors=couleurs, linewidths=np.where(m == 0, 3, 1))
    ax.add_collection(fronts)                               # d'abord : add_collection impose sinon le clip des axes entiers
    fronts.set_clip_path(Rectangle((-9.5, 0), 19, 6, transform=ax.transData))

    # sources secondaires : chacune émet une ondelette dès que le front l'atteint
    xs = np.arange(X0, X1 + 0.01, 0.6)
    thit = (xs - X0) * s1 / v1
    ax.plot(xs, np.zeros_like(xs), ".", color="0.3", ms=4)
    r = tau - thit
    ok = r > 0
    phi = np.linspace(0, np.pi, 60)

    def demi_cercles(vit, signe):
        rr = (vit * r[ok])[:, None]
        return np.stack([xs[ok][:, None] + rr * np.cos(phi)[None, :], signe * rr * np.sin(phi)[None, :]], axis=-1)

    if ok.any():
        ax.add_collection(LineCollection(demi_cercles(v2, -1), colors=to_rgba("C1", 0.8), linewidths=0.9))
        if refl:
            ax.add_collection(LineCollection(demi_cercles(v1, 1), colors=to_rgba("C4", 0.6), linewidths=0.9))

    def enveloppe(d, vit, couleur):
        """Droite tangente à toutes les ondelettes, tracée entre les ondelettes extrêmes."""
        r0 = vit * tau
        if s1 < 1e-9:                                      # incidence normale : enveloppe parallèle à l'interface
            ax.plot([-9.5, 9.5], [d[1] * r0] * 2, color=couleur, lw=3)
            return
        xf = X0 + v1 * tau / s1                            # point de l'interface atteint par le front en ce moment
        p0 = np.array([X0 + r0 * d[0], r0 * d[1]])
        if xf <= X1:
            p1 = np.array([xf, 0.0])
        else:
            rl = vit * (tau - (X1 - X0) * s1 / v1)
            p1 = np.array([X1 + rl * d[0], rl * d[1]])
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=couleur, lw=3)

    if th2 is not None:
        enveloppe((np.sin(np.radians(th2)), -np.cos(np.radians(th2))), v2, "C1")
    else:
        ax.text(0, -3, "Réflexion totale : les ondelettes n’ont pas d’enveloppe commune\n(pas d’onde réfractée)",
                ha="center", fontsize=11)
    if refl:
        enveloppe((s1, c1), v1, "C4")
    ax.text(-8.8, 5.3, f"milieu 1 : $n_1$ = {n1:.2f}", fontsize=11)
    ax.text(-8.8, -5.5, f"milieu 2 : $n_2$ = {n2:.2f}", fontsize=11)
    ax.set_xlim(-9.5, 9.5)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Chaque point du front devient une source d’ondelettes ; leur enveloppe est le nouveau front", fontsize=12)


def fig_rayons(fig, n1, n2, theta):
    ax, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.2, 1]})
    th = np.radians(theta)
    th2 = angle_refracte(n1, n2, theta)
    ax.axhspan(-1.4, 0, color="C0", alpha=0.05 + 0.12 * (n2 - 1) / 1.5)
    ax.axhline(0, color="0.4", lw=2)
    ax.plot([0, 0], [-1.3, 1.3], ":", color="0.5")
    fl = lambda a, b, col, lw=2.5, al=1.0: ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=col, lw=lw, alpha=al))
    # rayon incident et son angle
    fl((-1.2 * np.sin(th), 1.2 * np.cos(th)), (0, 0), "C0")
    ax.add_patch(Arc((0, 0), 0.9, 0.9, theta1=90, theta2=90 + theta, color="C0"))
    ax.text(-0.45 * np.sin(th / 2) - 0.12, 0.5 * np.cos(th / 2) + 0.05, r"$\theta_1$", color="C0", ha="center")
    # rayon réfléchi (épaisseur : part réfléchie, d'après les formules de Fresnel, lumière non polarisée)
    R = reflexion(n1, n2, np.array([th]))[2][0]
    fl((0, 0), (1.2 * np.sin(th), 1.2 * np.cos(th)), "C4", lw=1 + 3 * R, al=max(0.3, R))
    ax.add_patch(Arc((0, 0), 0.9, 0.9, theta1=90 - theta, theta2=90, color="C4"))
    ax.text(0.45 * np.sin(th / 2) + 0.12, 0.5 * np.cos(th / 2) + 0.05, r"$\theta_r$", color="C4", ha="center")
    titre = rf"Réflexion : $\theta_r = \theta_1$ = {theta}°" + "\n"
    if th2 is not None:
        t2 = np.radians(th2)
        fl((0, 0), (1.2 * np.sin(t2), -1.2 * np.cos(t2)), "C1", lw=1 + 3 * (1 - R), al=max(0.3, 1 - R))
        ax.add_patch(Arc((0, 0), 0.9, 0.9, theta1=-90, theta2=th2 - 90, color="C1"))
        ax.text(0.5 * np.sin(t2 / 2) + 0.05, -0.5 * np.cos(t2 / 2) - 0.12, r"$\theta_2$", color="C1")
        titre += rf"Réfraction : $n_1 \sin\theta_1 = n_2 \sin\theta_2$, $\theta_2$ = {th2:.1f}°"
    else:
        thc = np.degrees(np.arcsin(n2 / n1))
        titre += rf"Réflexion totale ($\theta_1 > \theta_c$ = {thc:.1f}°)"
    ax.set_title(titre, fontsize=11)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")

    a = np.linspace(0, 89.9, 400)
    s2 = n1 * np.sin(np.radians(a)) / n2
    ax2.plot(a, np.where(s2 <= 1, np.degrees(np.arcsin(np.minimum(s2, 1))), np.nan), color="C0", lw=2)
    ax2.plot(a, a, ":", color="0.6", label="sans déviation")
    if th2 is not None:
        ax2.plot([theta], [th2], "o", color="C3", ms=9)
    if n1 > n2:
        thc = np.degrees(np.arcsin(n2 / n1))
        ax2.axvline(thc, ls="--", color="C1", label=rf"angle limite $\theta_c$ = {thc:.1f}°")
        ax2.axvspan(thc, 90, color="C1", alpha=0.1)
    ax2.set_xlim(0, 90)
    ax2.set_ylim(0, 90)
    ax2.set_xlabel(r"$\theta_1$ (°)")
    ax2.set_ylabel(r"$\theta_2$ (°)")
    ax2.set_title("Angle réfracté", fontsize=12)
    ax2.legend(loc="upper left", fontsize=9)
