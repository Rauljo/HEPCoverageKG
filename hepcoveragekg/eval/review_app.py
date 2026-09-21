"""The review sheet as a page he can actually click through.

The HTML + TSV pair asks a busy physicist to keep two files in sync, scroll a
spreadsheet to the right row, and type a word. That friction is the reason this
would sit unopened for a fortnight, and it is blocking the whole evaluation
stage. So: one page, three buttons per item, progress kept in the browser, and
one file handed back at the end.

Two things it does that the paper sheet could not.

**The blinding is enforced rather than requested.** In the HTML our verdict sat
behind a <details> with a note asking him not to peek. Here the reveal is
disabled until he has answered that item. His answers cannot be a copy of ours
because he cannot see ours yet.

**Nothing is lost if he stops.** Every click is written to localStorage
immediately, so closing the tab costs nothing, and he can hand back partial work
at any point -- the questions are independent, so a half-finished sheet is still
a usable measurement for the questions it covers.

The reveal text is base64'd in the source. Not security -- anyone determined can
read it -- just so an accidental View Source does not spoil the exercise.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

# The evidence is set in a serif: it is the PAPER's words, and it should not look
# like our interface chrome. Everything we say around it stays in the UI sans.
_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{
  --paper:#fbfbf9; --surface:#fff; --ink:#16181d; --muted:#6a6f76;
  --line:#e5e5e0; --accent:#3d5a80; --accent-soft:#eaf0f6;
  --yes:#2d6a4f; --no:#9b2226; --unsure:#8a5a00;
  --shadow:0 1px 2px rgba(20,23,28,.06), 0 4px 14px rgba(20,23,28,.05);
}
@media (prefers-color-scheme:dark){
  :root{
    --paper:#131417; --surface:#1a1c21; --ink:#e7e7e3; --muted:#9aa0a8;
    --line:#2b2e35; --accent:#8fb0d0; --accent-soft:#20262e;
    --yes:#6cc39a; --no:#e8807f; --unsure:#d7a84b;
    --shadow:0 1px 2px rgba(0,0,0,.3), 0 4px 14px rgba(0,0,0,.25);
  }
}
:root[data-theme="dark"]{
  --paper:#131417; --surface:#1a1c21; --ink:#e7e7e3; --muted:#9aa0a8;
  --line:#2b2e35; --accent:#8fb0d0; --accent-soft:#20262e;
  --yes:#6cc39a; --no:#e8807f; --unsure:#d7a84b;
  --shadow:0 1px 2px rgba(0,0,0,.3), 0 4px 14px rgba(0,0,0,.25);
}
:root[data-theme="light"]{
  --paper:#fbfbf9; --surface:#fff; --ink:#16181d; --muted:#6a6f76;
  --line:#e5e5e0; --accent:#3d5a80; --accent-soft:#eaf0f6;
  --yes:#2d6a4f; --no:#9b2226; --unsure:#8a5a00;
  --shadow:0 1px 2px rgba(20,23,28,.06), 0 4px 14px rgba(20,23,28,.05);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:54rem;margin:0 auto;padding:0 1.1rem 6rem}

/* --- sticky bar --------------------------------------------------------- */
header{
  position:sticky; top:0; z-index:20; background:var(--paper);
  border-bottom:1px solid var(--line); padding:.7rem 0 .55rem; margin-bottom:1.4rem;
}
.bar{max-width:54rem;margin:0 auto;padding:0 1.1rem;
     display:flex;align-items:center;gap:1rem;flex-wrap:wrap}
.count{font-variant-numeric:tabular-nums;font-weight:600;white-space:nowrap}
.count small{font-weight:400;color:var(--muted)}
.track{flex:1 1 12rem;height:7px;background:var(--line);border-radius:99px;overflow:hidden;min-width:8rem}
.fill{height:100%;width:0;background:var(--accent);border-radius:99px;transition:width .25s ease}
.saved{font-size:.8rem;color:var(--muted);white-space:nowrap}
button{font:inherit;cursor:pointer;border-radius:7px;border:1px solid var(--line);
       background:var(--surface);color:var(--ink);padding:.4rem .8rem}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.primary{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
@media (prefers-color-scheme:dark){.primary{color:#12151a}}
:root[data-theme="dark"] .primary{color:#12151a}

/* --- intro -------------------------------------------------------------- */
.intro{background:var(--surface);border:1px solid var(--line);border-radius:12px;
       padding:1.2rem 1.3rem;margin-bottom:2rem;box-shadow:var(--shadow)}
.intro h1{margin:.1rem 0 .7rem;font-size:1.32rem;line-height:1.25;text-wrap:balance}
.intro p{margin:.55rem 0}
.intro .keyk{margin-top:.9rem;font-size:.87rem;color:var(--muted)}
kbd{font:600 .8rem ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--accent-soft);
    border:1px solid var(--line);border-radius:4px;padding:.1rem .35rem}
.flag{border-left:3px solid var(--accent);padding-left:.85rem;background:var(--accent-soft);
      padding:.7rem .85rem;border-radius:0 7px 7px 0;margin:.9rem 0}

/* --- questions ---------------------------------------------------------- */
h2{margin:2.6rem 0 .2rem;font-size:1.05rem;letter-spacing:.06em;text-transform:uppercase;
   color:var(--muted)}
.qtext{background:var(--accent-soft);border-left:3px solid var(--accent);
       padding:.75rem .95rem;border-radius:0 7px 7px 0;margin:.55rem 0 1.2rem;
       font-weight:500;text-wrap:pretty}
.qprog{font-size:.82rem;color:var(--muted);font-variant-numeric:tabular-nums}

/* --- item cards --------------------------------------------------------- */
.item{background:var(--surface);border:1px solid var(--line);border-radius:11px;
      padding:1rem 1.1rem;margin:.85rem 0;box-shadow:var(--shadow);scroll-margin-top:5.5rem}
.item.here{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent-soft),var(--shadow)}
.item.done{opacity:.62}
.item.done:hover,.item.done:focus-within{opacity:1}
.meta{display:flex;gap:.6rem;align-items:baseline;font-size:.82rem;color:var(--muted);
      margin-bottom:.5rem;flex-wrap:wrap}
.pid{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink)}
.peritem{font-weight:500;margin:.1rem 0 .6rem;text-wrap:pretty}
.ev{font-family:Georgia,"Iowan Old Style",Cambria,"Times New Roman",serif;
    font-size:.97rem;line-height:1.62;background:var(--paper);border:1px solid var(--line);
    border-left:3px solid var(--muted);border-radius:0 7px 7px 0;
    padding:.6rem .85rem;margin:.4rem 0;overflow-x:auto}
.nolead{font-size:.87rem;color:var(--muted);font-style:italic;margin:.3rem 0 .1rem}
/* The seven value rows state a number and ask whether it is right. The claim
   has to read as OURS and separately from the question, or the two run into
   one another -- which is exactly how the first draft of this batch read. */
.weanswer{margin:.7rem 0 .35rem;padding:.6rem .85rem;background:var(--accent-soft);
          border-radius:7px;font-weight:500;text-wrap:pretty}
.weanswer b{letter-spacing:.03em;text-transform:uppercase;font-size:.8rem;
            color:var(--muted);display:block;margin-bottom:.2rem}
.weask{font-size:.9rem;color:var(--muted);margin:0 0 .2rem}
.acts{display:flex;gap:.5rem;margin-top:.8rem;flex-wrap:wrap;align-items:center}
.v{padding:.42rem 1.05rem;font-weight:600;border-width:1.5px}
.v[aria-pressed="true"][data-v="yes"]{background:var(--yes);border-color:var(--yes);color:#fff}
.v[aria-pressed="true"][data-v="no"]{background:var(--no);border-color:var(--no);color:#fff}
.v[aria-pressed="true"][data-v="unsure"]{background:var(--unsure);border-color:var(--unsure);color:#fff}
@media (prefers-color-scheme:dark){.v[aria-pressed="true"]{color:#12151a}}
:root[data-theme="dark"] .v[aria-pressed="true"]{color:#12151a}
.tiny{font-size:.83rem;padding:.35rem .7rem;color:var(--muted)}
.tiny[disabled]{opacity:.45;cursor:not-allowed}
textarea{width:100%;margin-top:.6rem;padding:.5rem .65rem;border:1px solid var(--line);
         border-radius:7px;background:var(--paper);color:var(--ink);font:inherit;
         font-size:.92rem;resize:vertical;min-height:3.2rem}
.reveal{margin-top:.7rem;padding:.65rem .85rem;border-radius:7px;background:var(--paper);
        border:1px solid var(--line);border-left:3px solid var(--muted);font-size:.9rem}
.reveal b{color:var(--ink)}
.r-yes{border-left-color:var(--yes)}.r-no{border-left-color:var(--no)}

/* --- done panel --------------------------------------------------------- */
.done-panel{position:fixed;inset:0;background:rgba(10,12,15,.55);display:none;
            align-items:center;justify-content:center;padding:1.2rem;z-index:50}
.done-panel.on{display:flex}
.card{background:var(--surface);border-radius:13px;padding:1.4rem;max-width:38rem;width:100%;
      box-shadow:0 12px 40px rgba(0,0,0,.3);max-height:88vh;overflow:auto}
.card h3{margin:.1rem 0 .6rem}
.warnbox{background:var(--accent-soft);border-left:3px solid var(--unsure);
         padding:.65rem .85rem;border-radius:0 7px 7px 0;font-size:.92rem;margin:.7rem 0}
.out{width:100%;min-height:11rem;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-reduced-motion:reduce){*{transition:none!important;scroll-behavior:auto!important}}
</style>
</head>
<body>
<header>
  <div class="bar">
    <span class="count"><span id="n">0</span><small>/__TOTAL__ answered</small></span>
    <span class="track"><span class="fill" id="fill"></span></span>
    <span class="saved" id="saved">saved in this browser</span>
    <button class="primary" id="finish">Finish &amp; get my answers file</button>
  </div>
</header>

<div class="wrap">
  <div class="intro">
    <h1>__TITLE__</h1>
    <p>For each item below: <b>does the quoted sentence show that this paper does the
      thing the question describes?</b></p>
    <div class="flag">
      <p style="margin:0"><b>Many of these will be papers your own list does not
      contain.</b> That is deliberate, and it is the part I most need you on. Your
      answers were computed by filtering the pilot export, so a paper we cite and you
      did not list is not automatically our error — it may equally be something that
      export missed. Please judge <b>the sentence against the paper</b>, not against
      your list.</p>
    </div>
    <p>Some items say <b>we found no evidence</b>. For those I list the closest
      sentences retrieved — if none of them shows it, answer <b>no</b>; if one plainly
      does, we missed it, which is just as useful to know.</p>
    <p>After you answer, a link appears showing <b>what our model concluded and why</b>.
      It stays locked until you have answered, so your verdict cannot be a copy of
      ours — that is the whole point of asking you.</p>
    <p><b>Nothing is lost if you stop.</b> Every click is saved in this browser as you
      go, and the questions are independent — a half-finished set is still a usable
      measurement for the questions it covers. Press <b>Finish &amp; get my answers
      file</b> whenever you like — it hands you a small file to email back to
      __RETURN_TO__.</p>
    __NOTE__
    <p class="keyk">Keyboard: <kbd>y</kbd> yes · <kbd>n</kbd> no · <kbd>u</kbd> unsure ·
      <kbd>j</kbd>/<kbd>k</kbd> or <kbd>↓</kbd>/<kbd>↑</kbd> move · <kbd>r</kbd> reveal ·
      <kbd>?</kbd> notes</p>
  </div>
  <main id="list"></main>
</div>

<div class="done-panel" id="panel">
  <div class="card">
    <h3 id="panel-title">Getting your answers back to __RETURN_TO__</h3>
    <p id="panel-msg">This is everything you have marked so far.</p>
    <p class="warnbox"><b>This page cannot send anything on its own</b> — it has no
      connection to __RETURN_TO__'s machine. Please download the file (or copy the text)
      and email it back, otherwise the answers stay on this computer.</p>
    <button class="primary" id="copy">Copy my answers</button>
    <button id="dl">Download as a file</button>
    <button id="close" style="float:right">Close</button>
    <p style="font-size:.85rem;color:var(--muted);margin:.9rem 0 .3rem">
      Either copy this straight into an email, or download it as a file — whichever is
      easier. Both contain exactly the same thing.</p>
    <textarea class="out" id="out" readonly></textarea>
  </div>
</div>

<script>
const ITEMS = __ITEMS__;
const KEY = "hepckg-review-__VERSION__";
const state = load();
let cursor = 0;

function load(){
  try{ return JSON.parse(localStorage.getItem(KEY) || "{}"); }
  catch(e){ return {}; }
}
function save(){
  try{
    localStorage.setItem(KEY, JSON.stringify(state));
    flash("saved");
  }catch(e){
    // A full or disabled localStorage must not fail silently: his work is the
    // only copy, and losing it without a word is the worst outcome here.
    document.getElementById("saved").textContent =
      "COULD NOT SAVE — please send your answers back before closing";
    document.getElementById("saved").style.color = "var(--no)";
  }
}
let flashT;
function flash(msg){
  const el = document.getElementById("saved");
  el.textContent = msg; el.style.color = "";
  clearTimeout(flashT);
  flashT = setTimeout(()=>{ el.textContent = "saved in this browser"; }, 1200);
}
const dec = s => decodeURIComponent(escape(atob(s)));

function esc(s){
  return String(s == null ? "" : s).replace(/[&<>"]/g, c =>
    ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
}

function render(){
  const list = document.getElementById("list");
  let html = "", lastQ = null;
  ITEMS.forEach((it, idx) => {
    if(it.qid !== lastQ){
      lastQ = it.qid;
      const grp = ITEMS.filter(x => x.qid === it.qid);
      html += `<h2>${esc(it.qid)} <span class="qprog" id="qp-${esc(it.qid)}"></span></h2>`;
      const shared = new Set(grp.map(x => x.question)).size === 1;
      if(shared) html += `<div class="qtext">${it.question}</div>`;
    }
    const shared = ITEMS.filter(x => x.qid === it.qid);
    const perItem = new Set(shared.map(x => x.question)).size !== 1;
    let ev = "";
    if(it.quotes && it.quotes.length){
      if(it.quotes.length > 1)
        ev += `<div class="nolead">${it.quotes.length} sentences bearing on this question:</div>`;
      ev += it.quotes.map(q => `<div class="ev">${q}</div>`).join("");
    }else if(it.candidates && it.candidates.length){
      ev += `<div class="nolead">We found no evidence. The closest sentences in this paper were:</div>`;
      ev += it.candidates.map(q => `<div class="ev">${q}</div>`).join("");
    }else{
      ev += `<div class="nolead">We found no evidence, and nothing in this paper looked close.</div>`;
    }
    html += `<article class="item" id="it-${it.row}" data-idx="${idx}">
      <div class="meta"><span>#${it.row}</span><span class="pid">${esc(it.paper_id)}</span></div>
      ${perItem ? `<div class="peritem">${it.question}</div>` : ""}
      ${it.our_answer ? `<div class="weanswer"><b>Our answer</b>${it.our_answer}</div>
         <div class="weask">Is that right? If not, the correct value in the notes.</div>` : ""}
      ${ev}
      <div class="acts">
        <button class="v" data-v="yes"    data-row="${it.row}">Yes</button>
        <button class="v" data-v="no"     data-row="${it.row}">No</button>
        <button class="v" data-v="unsure" data-row="${it.row}">Unsure</button>
        <button class="tiny note-btn" data-row="${it.row}">Add a note</button>
        <button class="tiny rev-btn" data-row="${it.row}" disabled
          title="Answer first — then you can see what our model said">What our model said</button>
      </div>
      <div class="note-wrap" data-row="${it.row}" hidden>
        <textarea placeholder="Anything worth recording — a caveat, a better sentence, why it is borderline"
          data-row="${it.row}"></textarea>
      </div>
      <div class="rev-wrap" data-row="${it.row}" hidden></div>
    </article>`;
  });
  list.innerHTML = html;
  ITEMS.forEach(it => paint(it.row));
  updateCounts();
  move(0);
}

function paint(row){
  const it = ITEMS.find(x => x.row === row);
  const st = state[row] || {};
  const card = document.getElementById("it-" + row);
  card.querySelectorAll(".v").forEach(b =>
    b.setAttribute("aria-pressed", String(b.dataset.v === st.v)));
  card.classList.toggle("done", !!st.v);
  const rev = card.querySelector(".rev-btn");
  rev.disabled = !st.v;
  if(st.note){
    card.querySelector(".note-wrap").hidden = false;
    card.querySelector("textarea").value = st.note;
  }
  if(st.revealed && st.v){
    const w = card.querySelector(".rev-wrap");
    w.hidden = false;
    const cls = it.judge === true ? "r-yes" : (it.judge === false ? "r-no" : "");
    w.innerHTML = `<div class="reveal ${cls}"><b>${esc(dec(it.verdict_b64))}</b><br>${esc(dec(it.why_b64))}</div>`;
  }
}

function updateCounts(){
  const n = ITEMS.filter(it => (state[it.row] || {}).v).length;
  document.getElementById("n").textContent = n;
  document.getElementById("fill").style.width = (100 * n / ITEMS.length) + "%";
  const qids = [...new Set(ITEMS.map(i => i.qid))];
  qids.forEach(q => {
    const grp = ITEMS.filter(i => i.qid === q);
    const d = grp.filter(i => (state[i.row] || {}).v).length;
    const el = document.getElementById("qp-" + q);
    if(el) el.textContent = `${d}/${grp.length}`;
  });
}

function setV(row, v){
  state[row] = Object.assign({}, state[row], {v});
  save(); paint(row); updateCounts();
}

function move(i){
  cursor = Math.max(0, Math.min(ITEMS.length - 1, i));
  document.querySelectorAll(".item.here").forEach(e => e.classList.remove("here"));
  const el = document.getElementById("it-" + ITEMS[cursor].row);
  el.classList.add("here");
  el.scrollIntoView({block:"center", behavior:"smooth"});
}

document.addEventListener("click", ev => {
  const v = ev.target.closest(".v");
  if(v){
    const row = +v.dataset.row;
    setV(row, v.dataset.v);
    const idx = ITEMS.findIndex(x => x.row === row);
    // Step on automatically: 202 items is a lot of scrolling otherwise.
    if(idx === cursor && idx < ITEMS.length - 1) setTimeout(()=>move(idx + 1), 130);
    else move(idx);
    return;
  }
  const n = ev.target.closest(".note-btn");
  if(n){
    const w = document.querySelector(`.note-wrap[data-row="${n.dataset.row}"]`);
    w.hidden = !w.hidden;
    if(!w.hidden) w.querySelector("textarea").focus();
    return;
  }
  const r = ev.target.closest(".rev-btn");
  if(r && !r.disabled){
    const row = +r.dataset.row;
    state[row] = Object.assign({}, state[row], {revealed:true});
    save(); paint(row);
    return;
  }
});

document.addEventListener("input", ev => {
  if(ev.target.matches("textarea[data-row]")){
    const row = +ev.target.dataset.row;
    state[row] = Object.assign({}, state[row], {note: ev.target.value});
    save();
  }
});

document.addEventListener("keydown", ev => {
  if(ev.target.matches("textarea, input") || ev.metaKey || ev.ctrlKey || ev.altKey) return;
  const row = ITEMS[cursor].row;
  const k = ev.key.toLowerCase();
  if(k === "y"){ setV(row, "yes"); move(cursor + 1); ev.preventDefault(); }
  else if(k === "n"){ setV(row, "no"); move(cursor + 1); ev.preventDefault(); }
  else if(k === "u"){ setV(row, "unsure"); move(cursor + 1); ev.preventDefault(); }
  else if(k === "j" || ev.key === "ArrowDown"){ move(cursor + 1); ev.preventDefault(); }
  else if(k === "k" || ev.key === "ArrowUp"){ move(cursor - 1); ev.preventDefault(); }
  else if(k === "r"){
    const b = document.querySelector(`.rev-btn[data-row="${row}"]`);
    if(b && !b.disabled) b.click();
    ev.preventDefault();
  }
  else if(k === "?" || k === "/"){
    const b = document.querySelector(`.note-btn[data-row="${row}"]`);
    if(b) b.click();
    ev.preventDefault();
  }
});

function payload(){
  // Tab-separated, not JSON. This text is going to be pasted into an email by a
  // human, and 202 rows of pretty-printed JSON is a wall; a table is legible,
  // survives a mail client mangling whitespace, and parses just as easily on
  // our side. The header line carries the sheet version so a returned file can
  // never be scored against the wrong build of the sheet.
  const done = ITEMS.filter(i => (state[i.row] || {}).v);
  const head = `# sheet=__VERSION__\tanswered=${done.length}\ttotal=${ITEMS.length}\treturned=${new Date().toISOString()}`;
  const cols = "row\tquestion_id\tpaper\tverdict\tnotes";
  const rows = done.map(i => [
    i.row, i.qid, i.paper_id, state[i.row].v,
    (state[i.row].note || "").replace(/[\t\r\n]+/g, " ")
  ].join("\t"));
  return [head, cols].concat(rows).join("\n");
}

const panel = document.getElementById("panel");
document.getElementById("finish").onclick = () => {
  const n = ITEMS.filter(i => (state[i.row] || {}).v).length;
  document.getElementById("panel-msg").textContent =
    n === ITEMS.length
      ? `All ${n} answered — thank you, this is exactly what we needed.`
      : `${n} of ${ITEMS.length} answered. Partial is genuinely useful: every question you finished is a usable measurement, and you can come back to the rest later — this page remembers where you were.`;
  document.getElementById("out").value = payload();
  panel.classList.add("on");
};
document.getElementById("close").onclick = () => panel.classList.remove("on");
panel.addEventListener("click", e => { if(e.target === panel) panel.classList.remove("on"); });

document.getElementById("dl").onclick = () => {
  // A plain Blob download, deliberately: declaring the downloads capability
  // stops the page being shared, and a page his supervisor cannot open is worth
  // nothing however nicely it saves files. Some sandboxes block this silently,
  // which is why the copy box below is always on screen rather than a fallback
  // he has to go looking for.
  const btn = document.getElementById("dl");
  try{
    const blob = new Blob([payload()], {type:"text/plain;charset=utf-8"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "review-answers.txt";
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(()=>URL.revokeObjectURL(url), 4000);
    btn.textContent = "Saved — now email it to __RETURN_TO__";
  }catch(err){
    btn.textContent = "Download blocked — please copy the text below instead";
  }
};
document.getElementById("copy").onclick = async () => {
  const ta = document.getElementById("out");
  try{
    await navigator.clipboard.writeText(ta.value);
    document.getElementById("copy").textContent = "Copied — now paste it into an email";
  }catch(e){
    ta.select();
    document.getElementById("copy").textContent = "Press Cmd/Ctrl+C";
  }
};

render();
const restored = Object.keys(state).filter(r => state[r].v).length;
if(restored) flash(`welcome back — ${restored} answers restored`);
</script>
</body>
</html>
"""


def _b64(text: str) -> str:
    return base64.b64encode((text or "").encode("utf-8")).decode("ascii")


def write_app(items: list[dict], path: Path | str, title: str,
              version: str = "v1", return_to: str = "the author",
              note: str = "") -> Path:
    """One self-contained page: click yes/no/unsure, get one file back.

    `return_to` is named in the page because the page cannot send anything. It
    has no network path back to us -- the download is a local file save that the
    viewer must accept -- so the last step is a human emailing a file. The first
    version of this called its button "Send answers back", which promised a
    transmission that does not exist: he would have clicked it, closed the panel,
    believed he was finished, and we would have waited on answers already sitting
    in his Downloads folder.
    """
    # THE PAPER'S MATHS, RENDERED. 70 of batch 2's 104 rows carry a maths span,
    # and raw LaTeX makes the reviewer do the typesetting himself before he can
    # judge anything: `$p_{\mathrm{T}}^{\text{miss}}$` instead of p_T^miss.
    # `latex_html.to_html` escapes the text FIRST and only then adds sub/sup, so
    # what comes back is safe to insert without escaping again -- which is why
    # the template drops `esc()` on these two fields and only these two.
    from .latex_html import looks_like_latex, to_html

    def render_field(value: str) -> str:
        """LaTeX gets converted; plain English gets escaped and left alone."""
        import html as _html
        text = value or ""
        if looks_like_latex(text):
            return to_html(text)
        # Newlines still have to survive: the value rows put our answer inside
        # the question, separated by blank lines.
        return _html.escape(text).replace("\n\n", "<br><br>").replace("\n", "<br>")

    payload = []
    for i in items:
        quotes = i.get("quotes") or ([i["quote"]] if i.get("quote") else [])
        quotes = [render_field(q) for q in quotes]
        if i["_judge"] is True:
            verdict = "Our model said this DOES answer the question."
        elif i["_judge"] is False:
            verdict = "Our model rejected this — it said the sentence does NOT answer the question."
        elif i.get("_by") == "split":
            verdict = ("Our readings of this passage disagreed with each other — "
                       "we could not settle this one.")
        elif i["_machine"]:
            verdict = "Our model cited this, and our judge never ruled on it."
        else:
            verdict = "Our model found no evidence here; the sentences above were retrieved for you."
        payload.append({
            "row": i["row"], "qid": i["qid"], "paper_id": i["paper_id"],
            "question": render_field(i["question"]), "quotes": quotes,
            # Only the value rows carry this. It is OUR claim, not the paper's
            # words, so it renders above the evidence and outside the serif.
            "our_answer": render_field(i["our_answer"]) if i.get("our_answer") else "",
            "candidates": [render_field(c) for c in (i.get("candidates") or [])],
            "judge": i["_judge"],
            # base64 so an accidental View Source does not spoil the blinding.
            # Not security -- anyone determined can decode it -- just a guard
            # against reading our answer by mistake before giving theirs.
            "verdict_b64": _b64(verdict),
            "why_b64": _b64(i.get("_why") or "(no reason recorded)"),
        })
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = (_TEMPLATE
            .replace("__ITEMS__", blob)
            .replace("__TOTAL__", str(len(items)))
            .replace("__VERSION__", version)
            .replace("__TITLE__", title)
            .replace("__RETURN_TO__", return_to)
            .replace("__NOTE__", note))
    path = Path(path)
    path.write_text(html, encoding="utf-8")
    return path
