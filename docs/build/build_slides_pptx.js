// docs/slides.md -> docs/slides.pptx : minimal white deck (black text, Arial), 16:9.
// Every bullet is its own paragraph, so PowerPoint can animate "by paragraph".
// Usage: NODE_PATH=<dir with pptxgenjs> node docs/build/build_slides_pptx.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const DOCS = path.join(__dirname, "..");
const md = fs.readFileSync(path.join(DOCS, "slides.md"), "utf8");

// ---- parse front matter + slides ----
const fm = {};
let body = md;
const m = md.match(/^---\n([\s\S]*?)\n---\n/);
if (m) {
  for (const line of m[1].split("\n")) {
    const kv = line.match(/^(\w+):\s*"(.*)"\s*$/);
    if (kv) fm[kv[1]] = kv[2];
  }
  body = md.slice(m[0].length);
}
const slides = [];
for (const chunk of body.split(/^# /m).slice(1)) {
  const [head, ...rest] = chunk.split("\n");
  slides.push({ title: head.trim(), lines: rest });
}

// "**bold** text" -> runs; strip markdown escapes and code ticks
function runs(text, base) {
  const clean = text.replace(/\\\+/g, "+").replace(/`/g, "");
  const out = [];
  clean.split(/(\*\*[^*]+\*\*)/).filter(Boolean).forEach((part) => {
    const bold = part.startsWith("**") && part.endsWith("**");
    out.push({ text: bold ? part.slice(2, -2) : part, options: { ...base, bold } });
  });
  return out;
}

function bodyRuns(lines) {
  const paras = [];
  for (const raw of lines) {
    if (!raw.trim() || raw.startsWith("```") || raw.startsWith("\\")) continue;
    let mm;
    if ((mm = raw.match(/^ {2,}[-*] (.*)/))) paras.push({ text: mm[1], opts: { bullet: { indent: 22 }, indentLevel: 1, fontSize: 20 } });
    else if ((mm = raw.match(/^[-*] (.*)/))) paras.push({ text: mm[1], opts: { bullet: { indent: 22 }, indentLevel: 0, fontSize: 24 } });
    else if ((mm = raw.match(/^\d+\. (.*)/))) paras.push({ text: mm[1], opts: { bullet: { type: "number", indent: 30 }, indentLevel: 0, fontSize: 24 } });
    else paras.push({ text: raw.trim(), opts: { bullet: false, indentLevel: 0, fontSize: 24, paraSpaceBefore: 10 } });
  }
  const out = [];
  paras.forEach((p, i) => {
    // paragraph-level options (bullet, indent, spacing) only on the first run, or
    // pptxgenjs starts a new paragraph at every run that carries them
    const { bullet, indentLevel, paraSpaceBefore, ...charOpts } = p.opts;
    const rs = runs(p.text, { ...charOpts, color: "000000" });
    Object.assign(rs[0].options, { bullet, indentLevel, paraSpaceAfter: 8, ...(paraSpaceBefore ? { paraSpaceBefore } : {}) });
    if (i < paras.length - 1) rs[rs.length - 1].options.breakLine = true;
    out.push(...rs);
  });
  return out;
}

// ---- deck ----
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.title = fm.title || "Alchemy";
pres.theme = { headFontFace: "Arial", bodyFontFace: "Arial" };
const W = 13.33, X = 0.8, CW = W - 2 * X;
const num = { x: W - 1.0, y: 6.95, w: 0.6, h: 0.35, fontSize: 12, color: "000000", align: "right" };

pres.defineSlideMaster({
  title: "Title", background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: X, y: 2.0, w: CW, h: 1.2, fontSize: 60, bold: true, color: "000000", align: "left", valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: X, y: 3.35, w: CW, h: 0.8, fontSize: 28, bold: true, color: "000000", valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "authors", type: "body", x: X, y: 4.9, w: CW, h: 0.5, fontSize: 18, color: "000000", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "track", type: "body", x: X, y: 5.45, w: CW, h: 0.5, fontSize: 18, color: "000000", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "Section", background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "label", type: "body", x: X, y: 2.5, w: CW, h: 0.6, fontSize: 24, color: "000000", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: X, y: 3.1, w: CW, h: 1.1, fontSize: 48, bold: true, color: "000000", align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "Content", background: { color: "FFFFFF" },
  slideNumber: num,
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: X, y: 0.55, w: CW, h: 1.0, fontSize: 40, bold: true, color: "000000", align: "left", valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: X, y: 1.85, w: CW, h: 4.9, fontSize: 24, color: "000000", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({ title: "Closing", background: { color: "FFFFFF" },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: X, y: 2.9, w: CW, h: 1.4, fontSize: 60, bold: true, color: "000000", align: "center", valign: "middle", margin: 0 }, text: "" } }] });

let section = "Opening";
pres.addSection({ title: section });
const t = pres.addSlide({ masterName: "Title", sectionTitle: section });
t.addText(fm.title || "Alchemy", { placeholder: "title" });
t.addText(fm.subtitle || "", { placeholder: "subtitle" });
t.addText(fm.author || "", { placeholder: "authors" });
t.addText(fm.date || "", { placeholder: "track" });

for (const s of slides) {
  const text = s.lines.join("\n");
  const part = text.match(/\\Large (Part \d+)\\par\}[\s\S]*?\\bfseries ([^\\]+)\\par/);
  if (part) {                          // section divider
    section = part[2].trim();
    pres.addSection({ title: section });
    const d = pres.addSlide({ masterName: "Section", sectionTitle: section });
    d.addText(part[1], { placeholder: "label" });
    d.addText(section, { placeholder: "title" });
    continue;
  }
  if (/Thank you/.test(text) && s.title.startsWith("{")) {
    const c = pres.addSlide({ masterName: "Closing", sectionTitle: section });
    c.addText("Thank you", { placeholder: "title" });
    continue;
  }
  const sl = pres.addSlide({ masterName: "Content", sectionTitle: section });
  sl.addText(s.title, { placeholder: "title" });
  sl.addText(bodyRuns(s.lines), { placeholder: "body" });
}

pres.writeFile({ fileName: path.join(DOCS, "slides.pptx") }).then((f) => {
  // drop ZIP directory entries (they break opening/co-editing from cloud drives)
  require("child_process").execSync(`zip -dq "${f}" '*/' || true`);
  console.log("wrote " + f);
});
