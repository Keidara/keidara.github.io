"""Réfraction : lois de Snell-Descartes, construction de Huygens-Fresnel et coefficients de Fresnel."""
import numpy as np
from matplotlib.patches import Arc
from kit import *

PARAMS = [
    Slider("n1", "Indice du milieu 1 (haut)", 1.0, 2.5, 1.0, step=0.01),
    Slider("n2", "Indice du milieu 2 (bas)", 1.0, 2.5, 1.5, step=0.01),
    Slider("theta", "Angle d’incidence θ₁", 0, 89, 40, unit="°"),
    Check("refl", "Montrer aussi les ondelettes réfléchies", False),
    Slider("t", "Avancement", 0.0, 1.0, 0.0, step=0.005, anim=12),
]

X0, X1 = -9.0, 9.0          # étendue des sources secondaires sur l'interface


def angle_refracte(n1, n2, theta):
    """Angle θ₂ en degrés, ou None s'il y a réflexion totale."""
    s2 = n1 * np.sin(np.radians(theta)) / n2
    return None if s2 > 1 else float(np.degrees(np.arcsin(s2)))


def fig_huygens(fig, n1, n2, theta, refl, t):
    ax = fig.subplots()
    th = np.radians(theta)
    s1, c1 = np.sin(th), np.cos(th)
    v1, v2 = 1.0, n1 / n2                                   # vitesses de l'onde (la fréquence est la même : λ₁ = 1)
    tau = 12 * t
    th2 = angle_refracte(n1, n2, theta)
    ax.axhspan(-6, 0, color="C0", alpha=0.05 + 0.12 * (n2 - 1) / 1.5)
    ax.axhline(0, color="0.4", lw=2)
    # fronts d'onde incidents (le front en gras est celui qui sert à la construction)
    u = np.linspace(-40, 40, 2000)
    for m in range(0, 10):
        s = X0 * s1 + v1 * tau - m
        x, y = s * s1 + u * c1, -s * c1 + u * s1
        ok = (y >= 0) & (y <= 6) & (np.abs(x) <= 9.5)
        ax.plot(x[ok], y[ok], color="C0", lw=3 if m == 0 else 1, alpha=1 if m == 0 else 0.5)
    # sources secondaires : chacune émet une ondelette dès que le front l'atteint
    xs = np.arange(X0, X1 + 0.01, 0.6)
    thit = (xs - X0) * s1 / v1
    ax.plot(xs, np.zeros_like(xs), ".", color="0.3", ms=4)
    phi = np.linspace(0, np.pi, 80)
    for xi, th_i in zip(xs, thit):
        r = tau - th_i
        if r <= 0:
            continue
        ax.plot(xi + v2 * r * np.cos(phi), -v2 * r * np.sin(phi), color="C1", lw=0.9, alpha=0.8)
        if refl:
            ax.plot(xi + v1 * r * np.cos(phi), v1 * r * np.sin(phi), color="C4", lw=0.9, alpha=0.6)

    def enveloppe(d, vit, couleur):
        """Droite tangente à toutes les ondelettes : p·d = x_front · (d_x), tracée entre les ondelettes extrêmes."""
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
        ax.text(0, -3, "Réflexion totale : les ondelettes n’ont pas d’enveloppe commune\n(pas d’onde réfractée)", ha="center", fontsize=11)
    if refl:
        enveloppe((s1, c1), v1, "C4")
    ax.text(-8.8, 5.3, f"milieu 1 : n₁ = {n1:.2f}", fontsize=11)
    ax.text(-8.8, -5.5, f"milieu 2 : n₂ = {n2:.2f}", fontsize=11)
    ax.set_xlim(-9.5, 9.5)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Chaque point atteint par le front devient une source d’ondelettes ; leur enveloppe est le nouveau front")


def fig_rayons(fig, n1, n2, theta):
    ax, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.2, 1]})
    th = np.radians(theta)
    th2 = angle_refracte(n1, n2, theta)
    ax.axhspan(-1.4, 0, color="C0", alpha=0.05 + 0.12 * (n2 - 1) / 1.5)
    ax.axhline(0, color="0.4", lw=2)
    ax.plot([0, 0], [-1.3, 1.3], ":", color="0.5")
    fl = lambda a, b, col, lw=2.5, al=1.0: ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=col, lw=lw, alpha=al))
    fl((-1.2 * np.sin(th), 1.2 * np.cos(th)), (0, 0), "C0")
    ax.add_patch(Arc((0, 0), 0.9, 0.9, theta1=90, theta2=90 + theta, color="C0"))
    ax.text(-0.45 * np.sin(th / 2) - 0.12, 0.5 * np.cos(th / 2) + 0.05, "θ₁", color="C0", ha="center")
    # réflexion partielle (formules de Fresnel, lumière non polarisée)
    R = reflexion(n1, n2, np.array([th]))[2][0]
    fl((0, 0), (1.2 * np.sin(th), 1.2 * np.cos(th)), "C4", lw=1 + 3 * R, al=max(0.3, R))
    ax.text(0.5 * np.sin(th) + 0.1, 0.6 * np.cos(th) + 0.1, "θ₁", color="C4")
    if th2 is not None:
        t2 = np.radians(th2)
        fl((0, 0), (1.2 * np.sin(t2), -1.2 * np.cos(t2)), "C1", lw=1 + 3 * (1 - R), al=max(0.3, 1 - R))
        ax.add_patch(Arc((0, 0), 0.9, 0.9, theta1=-90, theta2=th2 - 90, color="C1"))
        ax.text(0.5 * np.sin(t2 / 2) + 0.05, -0.5 * np.cos(t2 / 2) - 0.12, "θ₂", color="C1")
        ax.set_title(f"n₁ sin θ₁ = n₂ sin θ₂ : θ₂ = {th2:.1f}°", fontsize=12)
    else:
        thc = np.degrees(np.arcsin(n2 / n1))
        ax.set_title(f"Réflexion totale (θ₁ > θc = {thc:.1f}°)", fontsize=12)
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
        ax2.axvline(thc, ls="--", color="C1", label=f"angle limite {thc:.1f}°")
        ax2.axvspan(thc, 90, color="C1", alpha=0.1)
    ax2.set_xlim(0, 90)
    ax2.set_ylim(0, 90)
    ax2.set_xlabel("θ₁ (°)")
    ax2.set_ylabel("θ₂ (°)")
    ax2.set_title("Angle réfracté", fontsize=12)
    ax2.legend(loc="upper left", fontsize=9)


def reflexion(n1, n2, th):
    """Facteurs de réflexion en intensité (polarisation s, polarisation p, moyenne), pour des angles th en radians."""
    th = np.asarray(th, float)
    s2 = n1 * np.sin(th) / n2
    tir = s2 > 1
    c2 = np.sqrt(1 - np.minimum(s2, 1) ** 2)
    c1 = np.cos(th)
    rs = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)
    rp = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)
    Rs, Rp = np.where(tir, 1.0, rs ** 2), np.where(tir, 1.0, rp ** 2)
    return Rs, Rp, (Rs + Rp) / 2


def fig_fresnel(fig, n1, n2, theta):
    ax = fig.subplots()
    a = np.linspace(0, 89.9, 600)
    Rs, Rp, Rm = reflexion(n1, n2, np.radians(a))
    ax.plot(a, 100 * Rs, color="C0", lw=2, label="polarisation s (perpendiculaire au plan d’incidence)")
    ax.plot(a, 100 * Rp, color="C1", lw=2, label="polarisation p (dans le plan d’incidence)")
    ax.plot(a, 100 * Rm, "--", color="0.5", lw=1.5, label="lumière non polarisée")
    r0 = float(reflexion(n1, n2, np.radians([float(theta)]))[2][0])
    ax.plot([theta], [100 * r0], "o", color="C3", ms=10)
    if abs(n1 - n2) > 1e-9:
        ab = np.degrees(np.arctan(n2 / n1))
        ax.axvline(ab, ls=":", color="C1")
        ax.text(ab, 60, f" angle de Brewster {ab:.1f}°", color="C1", rotation=90, va="center", ha="right", fontsize=9)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Angle d’incidence θ₁ (°)")
    ax.set_ylabel("Lumière réfléchie (%)")
    ax.set_title(f"θ₁ = {theta}° : {100 * r0:.1f} % réfléchi, {100 * (1 - r0):.1f} % transmis (lumière non polarisée)")
    ax.legend(loc="center left", fontsize=9)
