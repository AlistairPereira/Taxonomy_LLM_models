# app.py
# Minimal taxonomy inference web app (Flask + D3)
# - Enter Title/Description in the form
# - Backend assigns Cat1->Cat2->Cat3 using your trained artifacts
# - Frontend draws a small top-down path tree with D3.js

import os, re, html, json
import numpy as np
import pandas as pd
from collections import defaultdict

from flask import Flask, request, jsonify, render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

from sentence_transformers import SentenceTransformer
import pickle

# ---------- Paths to your artifacts ----------
PH4_C1 = "phase4_centroids_cat1.npy"
PH4_C2 = "phase4_centroids_cat2.npy"
PH4_C3 = "phase4_centroids_cat3.npy"
PH4_ASS = "phase4_clusters.parquet"
PH5_SUM = "phase5_node_summary.csv"
PH3_TAB = "phase3_table.parquet"
VEC_PKL = "phase4_tfidf_vectorizer.pkl"  # optional but recommended

# ---------- Load artifacts (once) ----------
if not (os.path.exists(PH4_C1) and os.path.exists(PH4_C2) and os.path.exists(PH4_C3)):
    raise FileNotFoundError("Missing Phase 4 centroid files. Ensure phase4_centroids_cat*.npy exist.")

C1_cent = np.load(PH4_C1, mmap_mode="r")  # SBERT space, L2 unit (K1, 384)
C2_cent = np.load(PH4_C2, mmap_mode="r")  # TF-IDF space, L2 unit (K2, V)
C3_cent = np.load(PH4_C3, mmap_mode="r")  # TF-IDF space, L2 unit (K3, V)

assign = pd.read_parquet(PH4_ASS)
nodes  = pd.read_csv(PH5_SUM)  # columns: level,node_id,label,size,top_terms,examples

# label maps
lab_c1 = {int(r.node_id): r.label for _, r in nodes[nodes.level=="cat1"].iterrows()}
lab_c2 = {int(r.node_id): r.label for _, r in nodes[nodes.level=="cat2"].iterrows()}
lab_c3 = {int(r.node_id): r.label for _, r in nodes[nodes.level=="cat3"].iterrows()}

# parent→children constraints (from TRAIN)
train_assign = assign[assign["split"]=="train"].copy()
C1_to_C2 = defaultdict(set)
C2_to_C3 = defaultdict(set)
for _, r in train_assign.iterrows():
    C1_to_C2[int(r.cat1)].add(int(r.cat2))
    C2_to_C3[int(r.cat2)].add(int(r.cat3))
C1_to_C2 = {k: sorted(list(v)) for k,v in C1_to_C2.items()}
C2_to_C3 = {k: sorted(list(v)) for k,v in C2_to_C3.items()}

# TF-IDF vectorizer
if os.path.exists(VEC_PKL):
    with open(VEC_PKL, "rb") as f:
        vec = pickle.load(f)
else:
    if not os.path.exists(PH3_TAB):
        raise FileNotFoundError(
            "Missing TF-IDF vectorizer pickle and no fallback data. "
            "Either provide phase4_tfidf_vectorizer.pkl or phase3_table.parquet (to refit)."
        )
    df_train_text = pd.read_parquet(PH3_TAB)
    df_train_text = df_train_text[df_train_text["split"]=="train"].copy()
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.7, stop_words="english")
    vec.fit(df_train_text["cleaned_text"].fillna(""))

# SBERT model (ensure the model is available or cached locally)
sbert = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Cleaning ----------
STOP = set("""
the and of for with to in on at a an by from into over under up down off as is are be
""".split())
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+._/-]*")

def clean_text(title: str, desc: str = "") -> str:
    s = f"{title or ''} {desc or ''}"
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"http\\S+|www\\.\\S+", " ", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9+\\-_/ ]+", " ", s)
    s = re.sub(r"\\s+", " ", s).strip()
    toks = TOKEN_RE.findall(s)
    toks = [t for t in toks if t not in STOP]
    return " ".join(toks)

# ---------- Inference ----------
def assign_one(title: str, description: str = ""):
    text = clean_text(title, description)

    # Cat1 (SBERT space)
    emb = sbert.encode([text], show_progress_bar=False)
    emb = sk_normalize(emb)  # (1, 384)
    c1 = int(np.argmax(emb @ C1_cent.T))

    # Cat2 (TF-IDF space; restricted by Cat1)
    Xtf = vec.transform([text])           # csr (1, V)
    cand_c2 = C1_to_C2.get(c1, [])
    if cand_c2:
        v = sk_normalize(Xtf, norm="l2")
        score = (v @ C2_cent[cand_c2].T).A1
        c2 = int(cand_c2[int(np.argmax(score))])
    else:
        c2 = -1

    # Cat3 (TF-IDF; restricted by Cat2)
    cand_c3 = C2_to_C3.get(c2, [])
    if cand_c3:
        v = sk_normalize(Xtf, norm="l2")
        score = (v @ C3_cent[cand_c3].T).A1
        c3 = int(cand_c3[int(np.argmax(score))])
    else:
        c3 = -1

    return {
        "cat1": c1, "cat1_label": lab_c1.get(c1, f"C1:{c1}"),
        "cat2": c2, "cat2_label": lab_c2.get(c2, f"C2:{c2}"),
        "cat3": c3, "cat3_label": lab_c3.get(c3, f"C3:{c3}"),
    }

# ---------- Flask app ----------
app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Taxonomy Classifier</title>
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <style>
    body { font: 15px/1.4 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color:#222; }
    .card { max-width: 900px; margin: 0 auto; padding: 20px; border: 1px solid #e5e5e5; border-radius: 12px; box-shadow: 0 4px 18px rgba(0,0,0,.05); }
    h1 { margin-top: 0; font-size: 20px; }
    label { display:block; font-weight:600; margin: 10px 0 4px; }
    input, textarea { width:100%; padding:10px; border:1px solid #ddd; border-radius:8px; }
    button { margin-top: 12px; padding:10px 16px; border:0; border-radius:10px; background:#111; color:#fff; cursor:pointer; }
    .row { display:flex; gap:16px; }
    .col { flex:1; }
    .path { margin-top: 16px; padding:10px; background:#f7f7f8; border-radius:8px; }
    #tree { margin-top:18px; border:1px dashed #ddd; border-radius:10px; padding:10px; min-height: 220px; }
    .chip { display:inline-block; padding:4px 8px; margin-right:8px; border-radius:999px; background:#eef2ff; border:1px solid #c7d2fe; }
    .small { color:#666; font-size: 12px; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Taxonomy Classifier (Cat1 → Cat2 → Cat3)</h1>
    <div class="row">
      <div class="col">
        <label for="title">Title</label>
        <input id="title" placeholder="e.g., HP Gaming Laptop RTX3060"/>
      </div>
      <div class="col">
        <label for="desc">Description <span class="small">(optional)</span></label>
        <input id="desc" placeholder="15.6\" i7, 16GB RAM, SSD, RTX3060"/>
      </div>
    </div>
    <button id="btn">Classify</button>

    <div id="result" style="display:none">
      <div class="path">
        <strong>Predicted path:</strong>
        <span id="crumbs"></span>
      </div>
      <div id="tree"></div>
    </div>
  </div>

<script>
const btn = document.getElementById('btn');
btn.addEventListener('click', async () => {
  const title = document.getElementById('title').value || "";
  const description = document.getElementById('desc').value || "";
  if (!title.trim()) {
    alert("Please enter a Title");
    return;
  }
  const res = await fetch('/api/predict', {
    method:'POST',
    headers: { 'Content-Type':'application/json' },
    body: JSON.stringify({ title, description })
  });
  const data = await res.json();
  renderResult(data);
});

function renderResult(data) {
  const r = document.getElementById('result');
  r.style.display = 'block';
  const crumbs = document.getElementById('crumbs');
  crumbs.innerHTML = '';

  const parts = [
    {lvl:'Cat1', id:data.cat1, label:data.cat1_label},
    {lvl:'Cat2', id:data.cat2, label:data.cat2_label},
    {lvl:'Cat3', id:data.cat3, label:data.cat3_label},
  ];

  for (const p of parts) {
    const span = document.createElement('span');
    span.className = 'chip';
    span.textContent = `${p.lvl}:${p.id} ${p.label ? '('+p.label+')' : ''}`;
    crumbs.appendChild(span);
  }

  // Build tiny path tree data
  const treeData = {
    name: "All Products",
    children: [{
      name: `${parts[0].label || 'C1:'+parts[0].id}`,
      children: [{
        name: `${parts[1].label || 'C2:'+parts[1].id}`,
        children: [{
          name: `${parts[2].label || 'C3:'+parts[2].id}`
        }]
      }]
    }]
  };

  drawTree(treeData);
}

function drawTree(data) {
  const container = d3.select("#tree");
  container.selectAll("*").remove();

  const width = Math.min(800, container.node().clientWidth);
  const dx = 24, dy = 180;
  const tree = d3.tree().nodeSize([dx, dy]);
  const diagonal = d3.linkHorizontal().x(d => d.y).y(d => d.x);
  const root = d3.hierarchy(data);

  tree(root);

  let x0 = Infinity;
  let x1 = -x0;
  root.each(d => { if (d.x > x1) x1 = d.x; if (d.x < x0) x0 = d.x; });

  const svg = container.append("svg")
      .attr("viewBox", [0, 0, width, x1 - x0 + dx*4])
      .style("width", "100%")
      .style("height", `${x1 - x0 + dx*4}px`);

  const g = svg.append("g")
      .attr("transform", `translate(20,${dx - x0})`);

  g.append("g")
    .attr("fill", "none")
    .attr("stroke", "#999")
    .attr("stroke-opacity", 0.6)
    .attr("stroke-width", 1.2)
    .selectAll("path")
    .data(root.links())
    .join("path")
    .attr("d", diagonal);

  const node = g.append("g")
    .attr("stroke-linejoin", "round")
    .attr("stroke-width", 1.5)
    .selectAll("g")
    .data(root.descendants())
    .join("g")
    .attr("transform", d => `translate(${d.y},${d.x})`);

  node.append("circle")
    .attr("fill", d => d.depth === 0 ? "#4C78A8" : (d.depth === 1 ? "#72B7B2" : (d.depth === 2 ? "#F58518" : "#E45756")))
    .attr("r", 6)
    .attr("stroke", "#333");

  node.append("text")
    .attr("dy", "0.32em")
    .attr("x", d => d.children ? -10 : 10)
    .attr("text-anchor", d => d.children ? "end" : "start")
    .text(d => d.data.name)
    .clone(true).lower()
    .attr("stroke", "white")
    .attr("stroke-width", 3);
}
</script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True) or {}
    title = data.get("title", "")
    desc  = data.get("description", "")
    pred = assign_one(title, desc)
    return jsonify(pred)

if __name__ == "__main__":
    # Tip: set host="0.0.0.0" if you want to access from another device on LAN
    app.run(host="127.0.0.1", port=5000, debug=True)
