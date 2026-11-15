// Moteur commun des outils : tu n'as pas besoin de l'ouvrir.
// Il charge Python, lit PARAMS et les fonctions fig_xxx() du fichier .py de l'outil, puis remplit les
// emplacements marqués dans la page HTML : data-controles="a b" (contrôles), data-figure="xxx" (figure),
// data-simulation="xxx" (canvas animé) avec data-lecture / data-reinit / data-barre / data-stats,
// data-action="nom" (bouton), data-etat (zone de message).
const MOTEUR = document.currentScript.src;
const $$ = s => [...document.querySelectorAll(s)];
const outil = document.body.dataset.outil || "";
// Mode administrateur : ajoute ?debug à l'adresse (ou localStorage.debug = "1" dans la console)
const debug = new URLSearchParams(location.search).has("debug") || localStorage.getItem("debug") === "1";
let kit, params = {}, figs = [], sims = [], demandes = new Set(), derniereFig = 0, timer, attente = new Set(), enCours = null, dernier = 0;
const dark = () => matchMedia("(prefers-color-scheme: dark)").matches;
const etat = t => { const e = document.querySelector("[data-etat]"); if (e) e.textContent = t; };
if (debug) etat("Chargement de Python et de matplotlib…");

// ---- valeurs des paramètres (un paramètre sans contrôle dans la page garde sa valeur par défaut)
function lire(p) {
  const el = document.getElementById("p_" + p.name);
  if (p.kind === "check") return el ? el.checked : p.val;
  if (p.kind === "select") {
    const r = document.querySelector(`input[name="p_${p.name}"]:checked`);
    return r ? r.value : el ? el.value : p.val;
  }
  if (!el) return p.val;
  const s = +el.value;                                      // curseur : position -> valeur
  return p.log ? p.lo * Math.pow(p.hi / p.lo, s / 1000) : s;
}
const valeurs = () => Object.fromEntries(Object.values(params).map(p => [p.name, lire(p)]));

function montrer(p) {                                       // texte à côté du curseur
  if (p.kind !== "slider") return;
  let v = lire(p);
  v = p.int ? Math.round(v) : v;
  document.getElementById("o_" + p.name).textContent = (p.int ? v.toLocaleString("fr-FR") : v.toPrecision(3)) + (p.unit ? " " + p.unit : "");
}

// ---- contrôles désactivés selon la valeur d'un autre paramètre (Slider(..., actif=("mode", ["Deux fentes"])))
function appliquerActifs() {
  for (const p of Object.values(params)) {
    const el = p.actif && document.getElementById("p_" + p.name);
    if (el) el.disabled = !p.actif[1].includes(lire(params[p.actif[0]]));
  }
}

// ---- création d'un contrôle
function creer(p, boite) {
  let l;
  if (p.kind === "select" && p.options.length <= 4) {       // peu de choix : pastilles exclusives
    l = document.createElement("div");
    l.className = "grp";
    l.innerHTML = `<span>${p.label}</span><div class="pills" role="radiogroup" aria-label="${p.label}">` +
      p.options.map(o => `<label><input type="radio" name="p_${p.name}" value="${o}"${o === p.val ? " checked" : ""}><span>${o}</span></label>`).join("") + "</div>";
  } else {
    l = document.createElement("label");
    if (p.kind === "slider") {
      const min = p.log ? 0 : p.lo, max = p.log ? 1000 : p.hi, step = p.log ? 1 : p.anim > 0 ? "any" : p.step;
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
  boite.append(l);
  l.addEventListener("input", () => { montrer(p); appliquerActifs(); planifier(p.name); });
  montrer(p);
  if (p.kind === "slider" && p.anim > 0) {                  // curseur animable : bouton Lecture/Pause
    const b = document.createElement("button");
    b.textContent = "▶ Lecture"; b.onclick = () => basculer(p, b);
    boite.append(b);
  }
}

// ---- dessin
const concernees = noms => figs.filter(f => f.uses.some(u => noms.includes(u)));
function dessiner(f, rapide = false) {
  const t0 = performance.now();
  try {
    f.img.src = "data:image/png;base64," + kit.render(f.nom, JSON.stringify(valeurs()), dark(), rapide, f.w, f.h);
    if (debug) etat(`${f.nom} : ${Math.round(performance.now() - t0)} ms`);
  } catch (e) {
    etat(debug ? `Erreur dans fig_${f.nom} : ` + String(e).split("\n").slice(-2).join(" ") : "Impossible de tracer une figure.");
    console.error(e);
  }
}
function planifier(nom) {                                   // évite de recalculer à chaque pixel
  attente.add(nom); clearTimeout(timer);
  timer = setTimeout(() => { const n = [...attente]; attente.clear(); concernees(n).forEach(f => dessiner(f));
    sims.filter(s => s.init.some(u => n.includes(u))).forEach(initialiser); }, 40);
}
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => kit && figs.forEach(f => dessiner(f)));

// ---- simulations animées (canvas calculé par Python)
function initialiser(s) {
  JSON.parse(kit.init(s.nom, JSON.stringify(valeurs()))).forEach(n => demandes.add(n));
  s.fini = false;
}
function boucle(now) {
  try {
    for (const s of sims) {
      const res = kit.frame(s.nom, JSON.stringify(valeurs()), s.lecture);
      const i = JSON.parse(kit.infos(s.nom));
      if (!s.image || s.c.width !== i.taille) { s.c.width = s.c.height = i.taille; s.image = s.ctx.createImageData(i.taille, i.taille); }
      const b = res.getBuffer("u8"); s.image.data.set(b.data); b.release(); res.destroy();
      s.ctx.putImageData(s.image, 0, 0);
      s.fini = i.fini;
      if (s.stats) s.stats.textContent = i.texte;
      if (s.bouton) s.bouton.textContent = i.fini ? "Rejouer" : s.lecture ? "Pause" : "Lecture";
      (i.figures || []).forEach(n => demandes.add(n));
    }
    if (demandes.size && now - derniereFig > 250) {         // figures liées (ex. écran) : au plus 4 fois par seconde
      figs.filter(f => demandes.has(f.nom)).forEach(f => dessiner(f));
      demandes.clear(); derniereFig = now;
    }
    requestAnimationFrame(boucle);
  } catch (e) {
    etat(debug ? "Erreur Python : " + String(e).split("\n").slice(-2).join(" ") : "Impossible de poursuivre la simulation.");
    console.error(e);
  }
}

// ---- animation : fait avancer le curseur animable, en boucle
function tick(now) {
  if (!enCours) return;
  const p = enCours, el = document.getElementById("p_" + p.name);
  const dt = dernier ? Math.min((now - dernier) / 1000, 0.25) : 0;
  dernier = now;
  let v = +el.value + dt * (p.hi - p.lo) / p.anim;
  if (v > p.hi) v = p.lo + (v - p.hi) % (p.hi - p.lo);
  el.value = v; montrer(p);
  concernees([p.name]).forEach(f => dessiner(f));           // dessiner(f, true) = basse définition, plus rapide
  requestAnimationFrame(tick);
}
function basculer(p, b) {
  if (enCours === p) { enCours = null; b.textContent = "▶ Lecture"; }
  else { enCours = p; dernier = 0; b.textContent = "⏸ Pause"; requestAnimationFrame(tick); }
}

// ---- démarrage
async function demarrer() {
  if (!/^[a-z0-9_]+$/.test(outil)) throw new Error("outil inconnu (data-outil)");
  const py = await loadPyodide();
  await py.loadPackage(["numpy", "matplotlib"]);
  py.FS.mkdir("/kitlib");           // (pas "/lib" : ce dossier existe déjà dans Pyodide)
  py.FS.writeFile("/kitlib/kit.py", await (await fetch(new URL("kit.py", MOTEUR))).text());
  py.runPython("import sys; sys.path.insert(0, '/kitlib')");
  kit = py.pyimport("kit");
  const rep = await fetch(`./${outil}.py`);
  if (!rep.ok) throw new Error(`${outil}.py introuvable`);
  kit.load(await rep.text());
  const info = JSON.parse(kit.describe());
  info.params.forEach(p => params[p.name] = p);

  $$("[data-controles]").forEach(boite => {
    const noms = boite.dataset.controles.trim() === "*" ? Object.keys(params) : boite.dataset.controles.trim().split(/\s+/);
    noms.forEach(n => { if (!params[n]) throw new Error(`paramètre inconnu : ${n}`); creer(params[n], boite); });
  });
  $$("[data-figure]").forEach(d => {
    const nom = d.dataset.figure, meta = info.figures[nom];
    if (!meta) throw new Error(`fig_${nom}() introuvable dans ${outil}.py`);
    const [w, h] = (d.dataset.taille || "10x6").split("x").map(Number);
    const img = new Image();
    img.alt = d.dataset.alt || "";
    d.append(img);
    const f = { nom, img, w, h, uses: meta.uses };
    figs.push(f);
    if (meta.click) {                                       // figure cliquable (ex. portes de Monty Hall)
      img.style.cursor = "pointer";
      img.addEventListener("click", e => {
        const r = img.getBoundingClientRect();
        kit.click(nom, (e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height, JSON.stringify(valeurs()), w, h);
        dessiner(f);
      });
    }
  });
  $$("[data-simulation]").forEach(c => {
    const nom = c.dataset.simulation, meta = info.simulations[nom];
    if (!meta) throw new Error(`init_${nom}() introuvable dans ${outil}.py`);
    const s = { nom, c, ctx: c.getContext("2d"), init: meta.init, image: null, fini: false,
                lecture: !matchMedia("(prefers-reduced-motion: reduce)").matches,
                stats: document.querySelector(`[data-stats="${nom}"]`), bouton: document.querySelector(`[data-lecture="${nom}"]`) };
    sims.push(s);
    if (s.bouton) s.bouton.onclick = () => { if (s.fini) { initialiser(s); s.lecture = true; } else s.lecture = !s.lecture; };
    $$(`[data-reinit="${nom}"]`).forEach(b => b.onclick = () => initialiser(s));
    const barre = document.querySelector(`[data-barre="${nom}"]`);
    if (barre && meta.barre) {                              // barre de couleur
      const lut = kit.barre(nom), lb = lut.getBuffer("u8");
      barre.width = 256; barre.height = 1;
      barre.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(lb.data), 256, 1), 0, 0);
      lb.release(); lut.destroy();
    }
    if (meta.souris) {                                      // pinceau : clic ou glissé
      c.style.cursor = "crosshair";
      const envoyer = e => { const r = c.getBoundingClientRect();
        kit.souris(nom, (e.clientX - r.left) / r.width, 1 - (e.clientY - r.top) / r.height, JSON.stringify(valeurs())); };
      c.addEventListener("pointerdown", e => { c.setPointerCapture(e.pointerId); envoyer(e); });
      c.addEventListener("pointermove", e => { if (e.buttons) envoyer(e); });
    }
    initialiser(s);
  });
  appliquerActifs();
  $$("[data-action]").forEach(b => b.addEventListener("click", () => {
    kit.action(b.dataset.action, JSON.stringify(valeurs()));
    const cibles = b.dataset.cible ? b.dataset.cible.split(/\s+/) : null;   // figures à redessiner (par défaut toutes)
    figs.filter(f => !cibles || cibles.includes(f.nom)).forEach(f => dessiner(f));
  }));
  figs.forEach(f => dessiner(f));
  if (sims.length) requestAnimationFrame(boucle);
}

demarrer().catch(err => {
  let detail = err && err.message;
  if (!detail) { try { detail = JSON.stringify(err); } catch (_) { detail = String(err); } }
  detail = String(detail).split("\n").slice(-3).join(" ");
  etat(debug ? "Impossible de charger l'outil : " + detail : "Impossible de charger l'outil. Recharge la page ou réessaie plus tard.");
  console.error(err);
});
