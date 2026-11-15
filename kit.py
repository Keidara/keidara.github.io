"""Socle commun des outils : déclaration des paramètres et rendu matplotlib.

Un outil (fichier outils/xxx.py) définit :
    TITLE   : titre de la page
    INTRO   : texte d'explication (paragraphes séparés par une ligne vide)
    PARAMS  : liste de Slider(...), Select(...), Check(...)
    draw(fig, **paramètres) : dessine dans la figure matplotlib `fig`
    RANDOM  : (facultatif) True pour afficher un bouton « Relancer »
"""
import base64, io, json
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
    _tool["ns"] = ns = {"__name__": "outil"}
    exec(code, ns)


def describe():
    ns = _tool["ns"]
    return json.dumps(dict(title=ns["TITLE"], intro=ns.get("INTRO", ""),
                           random=ns.get("RANDOM", False), params=ns["PARAMS"]))


def render(values_json, dark, rapide=False):
    """Appelle draw() et renvoie la figure en PNG encodé en base64."""
    ns = _tool["ns"]
    v = json.loads(values_json)
    for p in ns["PARAMS"]:
        if p["kind"] == "slider" and p["int"]:
            v[p["name"]] = int(round(v[p["name"]]))
    fg, grid = ("#e8eaee", "#3a4150") if dark else ("#15181d", "#d9dde3")
    couleurs = ["#7aa2ff" if dark else "#2563eb", "#e5484d", "#30a46c", "#f5a524"]
    rc = {"text.color": fg, "axes.labelcolor": fg, "axes.edgecolor": grid,
          "xtick.color": fg, "ytick.color": fg, "grid.color": grid,
          "axes.spines.top": False, "axes.spines.right": False,
          "legend.frameon": False, "font.size": 12, "axes.titlesize": 14,
          "axes.prop_cycle": cycler(color=couleurs)}
    with matplotlib.rc_context(rc):
        fig = Figure(figsize=(10, 6), dpi=75 if rapide else 110, constrained_layout=True)
        ns["draw"](fig, **v)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True)
    return base64.b64encode(buf.getvalue()).decode()
