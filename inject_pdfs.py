#!/usr/bin/env python3
"""
inject_pdfs.py
==============
Legge tutorial_fast.pdf e guida_rapida_modello.pdf dalla cartella corrente,
li codifica in base64 e genera guides.html.

WORKFLOW:
    1. Metti i due PDF nella stessa cartella di questo script
    2. python3 inject_pdfs.py
    3. git add guides.html && git commit -m "chore: update guides" && git push
       (index.html carica guides.html via fetch — funziona su GitHub Pages)
"""

import base64, os, sys, textwrap

PDF_FILES = [
    ("tutorial_fast.pdf",        "Tutorial fast: come usare il calcolatore"),
    ("guida_rapida_modello.pdf", "Guida al profilo di potenza omniPD"),
]

# ── 1. Leggi e codifica ────────────────────────────────────────────────────────
pdf_data = []
for filename, label in PDF_FILES:
    if not os.path.exists(filename):
        print(f"[WARN] {filename} non trovato — saltato.")
        continue
    with open(filename, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("utf-8")
    size_kb = os.path.getsize(filename) // 1024
    print(f"[OK]   {filename}  ({size_kb} KB)")
    pdf_data.append((filename, label, b64))

if not pdf_data:
    sys.exit("[ERR] Nessun PDF trovato. Metti i .pdf nella stessa cartella dello script.")

# ── 2. Costruisci i blocchi HTML ───────────────────────────────────────────────
def pdf_block(idx, filename, label, b64):
    return f"""\
  <div class="pdf-card">
    <button class="pdf-toggle" onclick="togglePdf({idx})" aria-expanded="false">
      <span class="pdf-icon">&#x1F4C4;</span>
      <span class="pdf-label">{label}</span>
      <span class="pdf-chevron" id="chev-{idx}">&#x25BC;</span>
    </button>
    <div class="pdf-body" id="body-{idx}" hidden>
      <iframe src="data:application/pdf;base64,{b64}"
              width="100%" height="640"
              title="{label}"></iframe>
      <a class="pdf-dl"
         href="data:application/pdf;base64,{b64}"
         download="{filename}">&#x2B07; Scarica &ldquo;{label}&rdquo;</a>
    </div>
  </div>"""

cards = "\n".join(pdf_block(i, fn, lbl, b64) for i, (fn, lbl, b64) in enumerate(pdf_data))

# ── 3. Template guides.html ────────────────────────────────────────────────────
HTML = """\
<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8">
  <title>Guide omniPD</title>
  <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;600&family=Crimson+Pro:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    /* Tutti gli stili sono scopati dentro #guidesContainer
       per non inquinare index.html quando vengono iniettati via fetch */

    #guidesContainer .pdf-card {
      background: #ffffff;
      border: 2px solid #e0e7ff;
      border-radius: 14px;
      margin-bottom: 14px;
      overflow: hidden;
      box-shadow: 0 4px 20px rgba(102,126,234,.13);
      transition: box-shadow .25s, transform .25s;
    }
    #guidesContainer .pdf-card:hover {
      box-shadow: 0 8px 28px rgba(102,126,234,.22);
      transform: translateY(-2px);
    }

    #guidesContainer .pdf-toggle {
      all: unset;
      box-sizing: border-box;
      display: flex;
      align-items: center;
      gap: 14px;
      width: 100%;
      padding: 18px 22px;
      cursor: pointer;
      font-family: 'Crimson Pro', serif;
      font-size: 1.15rem;
      font-weight: 600;
      color: #667eea;
      transition: background .2s;
      border-radius: 14px 14px 0 0;
    }
    #guidesContainer .pdf-toggle:hover { background: rgba(102,126,234,.06); }
    #guidesContainer .pdf-toggle:focus-visible { outline: 3px solid #667eea; outline-offset: -3px; }

    #guidesContainer .pdf-icon  { font-size: 1.35rem; flex-shrink: 0; }
    #guidesContainer .pdf-label { flex: 1; }
    #guidesContainer .pdf-chevron {
      font-size: .85rem;
      color: #764ba2;
      flex-shrink: 0;
      transition: transform .3s ease;
    }
    #guidesContainer .pdf-toggle[aria-expanded="true"] .pdf-chevron { transform: rotate(180deg); }

    #guidesContainer .pdf-body { padding: 0 20px 20px; }
    #guidesContainer .pdf-body[hidden] { display: none; }
    #guidesContainer .pdf-body:not([hidden]) { animation: guideSlideDown .3s ease-out; }

    @keyframes guideSlideDown {
      from { opacity: 0; transform: translateY(-8px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    #guidesContainer .pdf-body iframe {
      display: block;
      border: 1px solid #e0e7ff;
      border-radius: 10px;
      background: #f7f9fc;
    }

    #guidesContainer .pdf-dl {
      display: inline-block;
      margin-top: 12px;
      padding: 9px 20px;
      background: linear-gradient(135deg, #667eea, #764ba2);
      color: #fff;
      border-radius: 10px;
      text-decoration: none;
      font-family: 'Fira Code', monospace;
      font-size: .85rem;
      font-weight: 600;
      transition: opacity .2s, transform .2s;
    }
    #guidesContainer .pdf-dl:hover { opacity: .88; transform: translateY(-2px); }
  </style>
</head>
<body>
CARDS
  <script>
    function togglePdf(idx) {
      var body = document.getElementById('body-' + idx);
      var btn  = body.previousElementSibling;   // .pdf-toggle
      var open = body.hasAttribute('hidden');

      if (open) {
        body.removeAttribute('hidden');
        btn.setAttribute('aria-expanded', 'true');
      } else {
        body.setAttribute('hidden', '');
        btn.setAttribute('aria-expanded', 'false');
      }
    }
  </script>
</body>
</html>
"""

HTML = HTML.replace("CARDS\n", cards + "\n")

# ── 4. Scrivi il file ──────────────────────────────────────────────────────────
out = "guides.html"
with open(out, "w", encoding="utf-8") as fh:
    fh.write(HTML)

size_kb = os.path.getsize(out) // 1024
print(f"\n[OK]   Generato: {out}  ({size_kb} KB)")
print(textwrap.dedent("""
    Prossimi passi:
      git add guides.html
      git commit -m "chore: update guides"
      git push
    GitHub Pages servirà guides.html insieme a index.html — tutto funzionerà.
"""))