"""
prepare_data.py — Build MS MARCO embedding datasets for benchmark_2.py.

Downloads MS MARCO passage corpus from HuggingFace (8.8M passages, takes
first N_CORPUS), encodes with BAAI/bge-base-en-v1.5 (768-dim), then
PCA-reduces to 128 / 256 / 384 / 512 dimensions.

Output (in ./data/):
    embeddings_{dim}.npy   shape (N_CORPUS, dim)  float32
    queries_{dim}.npy      shape (N_QUERIES, dim)  float32
    manifest.csv           one row per dim — sizes, variance retained

Note: embeddings are stored as binary .npy (not CSV) because a 1M×768
float32 matrix as CSV would be ~20 GB of text and extremely slow to load.
The manifest.csv is a human-readable summary of what was created.

Usage:
    pip install sentence-transformers datasets scikit-learn
    python prepare_data.py
"""

import csv
import gc
import os

import numpy as np
from datasets import load_dataset
from sklearn.decomposition import PCA

# ── Config ─────────────────────────────────────────────────────────────────────
N_CORPUS  = 1_000_000
N_QUERIES = 1_000
DIMS      = [128, 256, 384, 512, 768]   # 768 saved as-is; others via PCA
MODEL     = "BAAI/bge-base-en-v1.5"    # 768-dim encoder, top RAG model on MTEB
OUTDIR    = "data"
SEED      = 42
BATCH     = 512

os.makedirs(OUTDIR, exist_ok=True)

# ── Device detection (handles broken CUDA drivers gracefully) ──────────────────
import torch
_device = "cpu"
try:
    if torch.cuda.is_available():
        torch.zeros(1).cuda()          # actually exercises the driver
        _device = "cuda"
except Exception:
    pass
print(f"Encoding device: {_device}"
      + ("  (GPU unavailable — CPU encoding will take longer)" if _device == "cpu" else ""))

# ── 1. Stream MS MARCO passage corpus ─────────────────────────────────────────
print(f"\nStreaming MS MARCO passage corpus (Tevatron/msmarco-passage-corpus)...")
ds = load_dataset("Tevatron/msmarco-passage-corpus", split="train", streaming=True)

n_need = N_CORPUS + N_QUERIES
texts  = []
for row in ds:
    text = (row.get("text") or row.get("passage") or "").strip()
    if text:
        texts.append(text)
    if len(texts) % 100_000 == 0 and len(texts) > 0:
        print(f"  {len(texts):>9,} / {n_need:,}", flush=True)
    if len(texts) >= n_need:
        break

print(f"  Collected {len(texts):,} passages")
corpus_texts = texts[:N_CORPUS]
query_texts  = texts[N_CORPUS : N_CORPUS + N_QUERIES]
del texts
gc.collect()

# ── 2. Encode with sentence-transformers ──────────────────────────────────────
from sentence_transformers import SentenceTransformer

print(f"\nLoading {MODEL}...")
model = SentenceTransformer(MODEL, device=_device)

print(f"Encoding {N_CORPUS:,} corpus passages...")
corpus_emb = model.encode(
    corpus_texts,
    batch_size=BATCH,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype(np.float32)
print(f"  shape: {corpus_emb.shape}   size: {corpus_emb.nbytes/1e9:.2f} GB")

print(f"Encoding {N_QUERIES:,} query passages...")
query_emb = model.encode(
    query_texts,
    batch_size=BATCH,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype(np.float32)

del model, corpus_texts, query_texts
gc.collect()

# ── 3. Save 768-dim as-is ─────────────────────────────────────────────────────
out_c768 = f"{OUTDIR}/embeddings_768.npy"
out_q768 = f"{OUTDIR}/queries_768.npy"
np.save(out_c768, corpus_emb)
np.save(out_q768, query_emb)
print(f"\nSaved {out_c768}  ({os.path.getsize(out_c768)/1e9:.2f} GB)")

manifest_rows = [{
    "dim":               768,
    "n_corpus":          N_CORPUS,
    "n_queries":         N_QUERIES,
    "corpus_size_gb":    round(os.path.getsize(out_c768) / 1e9, 3),
    "queries_size_mb":   round(os.path.getsize(out_q768) / 1e6, 1),
    "variance_retained": 1.0,
    "source":            "BAAI/bge-base-en-v1.5 (native 768-dim)",
}]

# ── 4. PCA-reduce to lower dims ────────────────────────────────────────────────
print("\nApplying PCA to create lower-dimensional variants...")
for dim in [512, 384, 256, 128]:
    print(f"  dim={dim} ... ", end="", flush=True)
    pca = PCA(n_components=dim, random_state=SEED, svd_solver="randomized")
    fit_idx = np.random.default_rng(SEED).choice(N_CORPUS, size=100_000, replace=False)
    pca.fit(corpus_emb[fit_idx])
    var = float(pca.explained_variance_ratio_.sum())

    c_red = pca.transform(corpus_emb).astype(np.float32)
    q_red = pca.transform(query_emb).astype(np.float32)

    out_c = f"{OUTDIR}/embeddings_{dim}.npy"
    out_q = f"{OUTDIR}/queries_{dim}.npy"
    np.save(out_c, c_red)
    np.save(out_q, q_red)
    print(f"done  (variance retained: {var:.4f})")

    manifest_rows.append({
        "dim":               dim,
        "n_corpus":          N_CORPUS,
        "n_queries":         N_QUERIES,
        "corpus_size_gb":    round(os.path.getsize(out_c) / 1e9, 3),
        "queries_size_mb":   round(os.path.getsize(out_q) / 1e6, 1),
        "variance_retained": round(var, 4),
        "source":            f"PCA from 768-dim (fit on 100K subset)",
    })

    del pca, c_red, q_red
    gc.collect()

# ── 5. Write manifest CSV ──────────────────────────────────────────────────────
manifest_path = f"{OUTDIR}/manifest.csv"
fields = ["dim", "n_corpus", "n_queries", "corpus_size_gb", "queries_size_mb",
          "variance_retained", "source"]
manifest_rows.sort(key=lambda r: r["dim"])

with open(manifest_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(manifest_rows)

print(f"\nManifest written to {manifest_path}")
print("\n── Summary ───────────────────────────────────────────────────────────")
print(f"{'dim':>6}  {'n_corpus':>10}  {'size_gb':>8}  {'variance':>9}  source")
print("-" * 70)
for r in manifest_rows:
    print(f"{r['dim']:>6}  {r['n_corpus']:>10,}  {r['corpus_size_gb']:>8.3f}  "
          f"{r['variance_retained']:>9.4f}  {r['source']}")

print("\nDone. Run  python benchmark_2.py  to use real data automatically.")
