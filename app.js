// Colle JavaScript commune à tous les outils : tu n'as pas besoin de l'ouvrir.
// Elle charge Python, lit PARAMS dans outils/<nom>.py, crée les contrôles,
// appelle draw() à chaque changement et affiche l'image obtenue.
const $ = id => document.getElementById(id);
const tool = new URLSearchParams(location.search).get("t") || "";
// Mode administrateur : ajoute ?debug à l'adresse (ou localStorage.debug = "1" dans la console)
const debug = new URLSearchParams(location.search).has("debug") || localStorage.getItem("debug") === "1";
let kit, params = [], timer, playing = false, last = 0;
if (debug) $("stats").textContent = "Chargement de Python et de matplotlib…";   // info réservée au mode debug

// Valeur d'un paramètre, lue dans son contrôle
function value(p) {
  if (p.kind === "check") return $("p_" + p.name).checked;
  if (p.kind === "select") {                                // pastilles (radio) ou menu
    const r = document.querySelector(`input[name="p_${p.name}"]:checked`);
    return r ? r.value : $("p_" + p.name).value;
  }
  const s = +$("p_" + p.name).value;                        // curseur : position -> valeur
  return p.log ? p.lo * Math.pow(p.hi / p.lo, s / 1000) : s;
}

function show(p) {                                          // texte à côté du curseur
  if (p.kind !== "slider") return;
  let v = value(p);
  v = p.int ? Math.round(v) : v;
  $("o_" + p.name).textContent = (p.int ? v.toLocaleString("fr-FR") : v.toPrecision(3)) + (p.unit ? " " + p.unit : "");
}

function build() {
  const box = $("controls");
  for (const p of params) {
    let l;
    if (p.kind === "select" && p.options.length <= 4) {     // peu de choix : pastilles exclusives
      l = document.createElement("div");
      l.className = "grp";
      l.innerHTML = `<span>${p.label}</span><div class="pills" role="radiogroup" aria-label="${p.label}">` +
        p.options.map(o => `<label><input type="radio" name="p_${p.name}" value="${o}"${o === p.val ? " checked" : ""}><span>${o}</span></label>`).join("") + "</div>";
    } else {
      l = document.createElement("label");
      if (p.kind === "slider") {
        const min = p.log ? 0 : p.lo, max = p.log ? 1000 : p.hi;
        const step = p.log ? 1 : p.anim > 0 ? "any" : p.step;
        const v0 = p.log ? 1000 * Math.log(p.val / p.lo) / Math.log(p.hi / p.lo) : p.val;
        l.innerHTML = `<span class="row">${p.label} <output id="o_${p.name}"></output></span>
          <input id="p_${p.name}" type="range" min="${min}" max="${max}" step="${step}" value="${v0}">`;
      } else if (p.kind === "select") {
        l.innerHTML = `${p.label}<select id="p_${p.name}">` +
          p.options.map(o => `<option${o === p.val ? " selected" : ""}>${o}</option>`).join("") + "</select>";
      } else {
        l.className = "check";
        l.innerHTML = `<input id="p_${p.name}" type="checkbox"${p.val ? " checked" : ""}> ${p.label}`;
      }
    }
    box.append(l);
    l.addEventListener("input", () => { show(p); schedule(); });
    show(p);
  }
  if (params.some(p => p.anim > 0)) {                       // bouton Lecture/Pause si un curseur est animable
    const b = document.createElement("button");
    b.id = "play"; b.textContent = "▶ Lecture"; b.onclick = toggle;
    box.prepend(b);
  }
}

// Animation : fait avancer le curseur animable, en boucle, et redessine en basse définition
function tick(now) {
  if (!playing) return;
  const p = params.find(q => q.anim > 0), el = $("p_" + p.name);
  const dt = last ? Math.min((now - last) / 1000, 0.25) : 0;
  last = now;
  let v = +el.value + dt * (p.hi - p.lo) / p.anim;
  if (v > p.hi) v = p.lo + (v - p.hi) % (p.hi - p.lo);
  el.value = v; show(p); draw();          // draw(true) = basse définition, plus rapide
  requestAnimationFrame(tick);
}
function toggle() {
  playing = !playing; last = 0;
  $("play").textContent = playing ? "⏸ Pause" : "▶ Lecture";
  if (playing) requestAnimationFrame(tick); else draw();    // à l'arrêt : image nette
}

const dark = () => matchMedia("(prefers-color-scheme: dark)").matches;
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => kit && draw());

function schedule() { clearTimeout(timer); timer = setTimeout(() => draw(), 40); }   // évite de recalculer à chaque pixel

function draw(rapide = false) {
  const v = {};
  for (const p of params) v[p.name] = value(p);
  const t0 = performance.now();
  try {
    $("fig").src = "data:image/png;base64," + kit.render(JSON.stringify(v), dark(), rapide);
    $("stats").textContent = debug ? `Calcul et tracé : ${Math.round(performance.now() - t0)} ms` : "";
  } catch (e) {
    $("stats").textContent = debug ? "Erreur Python : " + String(e).split("\n").slice(-2).join(" ")
                                   : "Impossible de tracer la figure.";
    console.error(e);
  }
}

async function start() {
  if (!/^[a-z0-9_]+$/.test(tool)) throw new Error("outil inconnu");
  const py = await loadPyodide();
  await py.loadPackage(["numpy", "matplotlib"]);
  py.FS.mkdir("/kitlib");           // (pas "/lib" : ce dossier existe déjà dans Pyodide)
  py.FS.writeFile("/kitlib/kit.py", await (await fetch("./kit.py")).text());
  py.runPython("import sys; sys.path.insert(0, '/kitlib')");
  kit = py.pyimport("kit");
  const rep = await fetch(`./outils/${tool}.py`);
  if (!rep.ok) throw new Error(`outils/${tool}.py introuvable`);
  kit.load(await rep.text());
  const info = JSON.parse(kit.describe());
  params = info.params;
  document.title = info.title;
  $("titre").textContent = info.title;
  $("note").innerHTML = info.intro.split("\n\n").map(t => `<p>${t}</p>`).join("");
  build();
  $("fig").alt = info.title;
  if (info.random) { $("again").hidden = false; $("again").onclick = () => { kit.reset(); draw(); }; }
  if (info.click) {                                         // outil cliquable (ex. portes de Monty Hall)
    $("fig").style.cursor = "pointer";
    $("fig").addEventListener("click", e => {
      const r = $("fig").getBoundingClientRect(), v = {};
      for (const p of params) v[p.name] = value(p);
      kit.click((e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height, JSON.stringify(v));
      draw();
    });
  }
  draw();
}

start().catch(err => {
  let detail = err && err.message;
  if (!detail) { try { detail = JSON.stringify(err); } catch (_) { detail = String(err); } }
  detail = String(detail).split("\n").slice(-3).join(" ");
  $("stats").textContent = debug
    ? "Impossible de charger l'outil : " + detail
    : "Impossible de charger l'outil. Recharge la page ou réessaie plus tard.";
  console.error(err);
});
