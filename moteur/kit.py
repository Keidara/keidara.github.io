"""Socle commun des outils : déclaration des paramètres et rendu matplotlib.

Un outil (outils/xxx/xxx.py) définit :
    PARAMS            : liste de Slider(...), Select(...), Check(...)
    fig_nom(fig, a, b): dessine la figure « nom » ; ses arguments (après fig) sont des paramètres de PARAMS.
                        Elle n'est redessinée que si l'un d'eux change.
    clic_nom(x, y, a) : (facultatif) appelée quand on clique sur la figure « nom » (x dans [0, largeur], y vers le haut)
    fonction()        : (facultatif) appelable par un bouton <button data-action="fonction"> de la page
La page HTML (outils/xxx/xxx.html) décide de l'emplacement des figures et des contrôles.
"""
import base64, inspect, io, json
import matplotlib
matplotlib.use("agg")
from cycler import cycler
from matplotlib.figure import Figure

__all__ = ["Slider", "Select", "Check"]
_tool = {}


def Slider(name, label, lo, hi, val, step=1, log=False, unit="", anim=0):
    """anim : durée (en secondes) d'un balayage complet si le curseur doit pouvoir s'animer."""
    est_entier = all(isinstance(x, int) for x in (lo, hi, step))
    return dict(kind="slider", name=name, label=label, lo=lo, hi=hi, val=val,
                step=step, log=log, unit=unit, anim=anim, int=est_entier)


def Select(name, label, options, val=None):
    return dict(kind="select", name=name, label=label, options=list(options),
                val=val if val is not None else options[0])


def Check(name, label, val=False):
    return dict(kind="check", name=name, label=label, val=val)


def load(code):
    """Exécute le fichier de l'outil."""
    _tool.clear()
    _tool["ns"] = {"__name__": "outil"}
    exec(code, _tool["ns"])


def _figures():
    return {n[4:]: f for n, f in _tool["ns"].items() if n.startswith("fig_") and callable(f)}


def _args(f, saut):
    """Noms des paramètres d'une fonction, sans les `saut` premiers (fig, ou x et y)."""
    return list(inspect.signature(f).parameters)[saut:]


def describe():
    ns = _tool["ns"]
    figs = {nom: dict(uses=_args(f, 1), click=callable(ns.get("clic_" + nom))) for nom, f in _figures().items()}
    return json.dumps(dict(params=ns.get("PARAMS", []), figures=figs))


def _valeurs(values_json):
    v = json.loads(values_json)
    for p in _tool["ns"].get("PARAMS", []):
        if p["kind"] == "slider" and p["int"]:
            v[p["name"]] = int(round(v[p["name"]]))
    return v


def render(nom, values_json, dark, rapide=False, w=10, h=6):
    """Appelle fig_<nom>() et renvoie la figure en PNG encodé en base64."""
    f, v = _figures()[nom], _valeurs(values_json)
    fg, grid = ("#e8eaee", "#3a4150") if dark else ("#15181d", "#d9dde3")
    couleurs = ["#7aa2ff" if dark else "#2563eb", "#e5484d", "#30a46c", "#f5a524"]
    rc = {"text.color": fg, "axes.labelcolor": fg, "axes.edgecolor": grid,
          "xtick.color": fg, "ytick.color": fg, "grid.color": grid,
          "axes.spines.top": False, "axes.spines.right": False,
          "legend.frameon": False, "font.size": 12, "axes.titlesize": 14,
          "axes.prop_cycle": cycler(color=couleurs)}
    with matplotlib.rc_context(rc):
        fig = Figure(figsize=(w, h), dpi=75 if rapide else 110, constrained_layout=True)
        f(fig, **{k: v[k] for k in _args(f, 1)})
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True)
    return base64.b64encode(buf.getvalue()).decode()


def click(nom, fx, fy, values_json, w=10, h=6):
    """Clic sur la figure (fx, fy : fractions de l'image, fy compté depuis le haut)."""
    f = _tool["ns"].get("clic_" + nom)
    if f:
        v = _valeurs(values_json)
        f(fx * w, (1 - fy) * h, **{k: v[k] for k in _args(f, 2)})


def action(nom):
    """Bouton <button data-action="nom"> : appelle la fonction nom() de l'outil si elle existe."""
    f = _tool["ns"].get(nom)
    if callable(f) and not nom.startswith("_"):
        f()
