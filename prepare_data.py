"""
prepare_data.py — Build real RAG embedding datasets for benchmark_2.py.

Downloads Wikipedia from HuggingFace, encodes 1M passages with
BAAI/bge-base-en-v1.5 (768-dim) on GPU, then PCA-reduces to
128 / 256 / 384 / 512 dimensions.

Output (in ./data/):
    embeddings_{dim}.npy   shape (1_000_000, dim)  float32
    queries_{dim}.npy      shape (1_000,     dim)  float32

Usage:
    pip install sentence-transformers datasets scikit-learn
    python prepare_data.py
"""

import os
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# ── Config ─────────────────────────────────────────────────────────────────────
N_CORPUS  = 1_000_000
N_QUERIES = 1_000
DIMS      = [128, 256, 384, 512, 768]   # 768 saved as-is; others via PCA
MODEL     = "BAAI/bge-base-en-v1.5"    # 768-dim, top RAG model on MTEB
OUTDIR    = "data"
SEED      = 42
BATCH     = 512                         # encode batch size; tune down if GPU OOMs

os.makedirs(OUTDIR, exist_ok=True)

# ── 1. Stream Wikipedia passages ───────────────────────────────────────────────
print("Streaming Wikipedia (wikimedia/wikipedia 20231101.en) — collecting passages...")
ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

n_need  = N_CORPUS + N_QUERIES
texts   = []
for row in ds:
    # Take up to 512 characters — realistic single-paragraph RAG chunk
    chunk = row["text"][:512].strip()
    if chunk:
        texts.append(chunk)
    if len(texts) >= n_need:
        break

print(f"  Collected {len(texts):,} passages")
corpus_texts = texts[:N_CORPUS]
query_texts  = texts[N_CORPUS : N_CORPUS + N_QUERIES]

# ── 2. Encode with sentence-transformers on GPU ────────────────────────────────
print(f"\nLoading {MODEL}...")
model = SentenceTransformer(MODEL, device="cuda")

print(f"Encoding {N_CORPUS:,} corpus passages  (expect ~5–15 min on L4)...")
corpus_emb = model.encode(
    corpus_texts,
    batch_size=BATCH,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,   # cosine similarity ready
).astype(np.float32)
print(f"  corpus_emb: {corpus_emb.shape}  {corpus_emb.nbytes/1e9:.2f} GB")

print(f"Encoding {N_QUERIES:,} query passages...")
query_emb = model.encode(
    query_texts,
    batch_size=BATCH,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype(np.float32)

# Save 768-dim as-is
out_c = f"{OUTDIR}/embeddings_768.npy"
out_q = f"{OUTDIR}/queries_768.npy"
np.save(out_c, corpus_emb)
np.save(out_q, query_emb)
print(f"  Saved {out_c}  ({os.path.getsize(out_c)/1e9:.2f} GB)")
print(f"  Saved {out_q}")

# ── 3. PCA-reduce to lower dims ────────────────────────────────────────────────
print("\nApplying PCA to create lower-dimensional variants...")
for dim in [512, 384, 256, 128]:
    print(f"  dim={dim} ... ", end="", flush=True)
    pca = PCA(n_components=dim, random_state=SEED, svd_solver="randomized")
    # Fit on a 100K subset — captures variance well and is much faster than 1M
    fit_idx = np.random.default_rng(SEED).choice(N_CORPUS, size=100_000, replace=False)
    pca.fit(corpus_emb[fit_idx])
    var_explained = pca.explained_variance_ratio_.sum()

    c_reduced = pca.transform(corpus_emb).astype(np.float32)
    q_reduced = pca.transform(query_emb).astype(np.float32)

    np.save(f"{OUTDIR}/embeddings_{dim}.npy", c_reduced)
    np.save(f"{OUTDIR}/queries_{dim}.npy",    q_reduced)
    print(f"done  (variance retained: {var_explained:.3f})")

# ── 4. Summary ─────────────────────────────────────────────────────────────────
print("\nFiles written:")
for dim in DIMS:
    cp = f"{OUTDIR}/embeddings_{dim}.npy"
    qp = f"{OUTDIR}/queries_{dim}.npy"
    print(f"  {cp:<32}  {os.path.getsize(cp)/1e9:.2f} GB")
    print(f"  {qp:<32}  {os.path.getsize(qp)/1e6:.1f} MB")

print("\nDone. Run  python benchmark_2.py  to use real data automatically.")
