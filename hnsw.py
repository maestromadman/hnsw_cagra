"""
hnsw.py — hnswlib (CPU HNSW) vector index wrapper.

Exposes a single `run()` function consumed by benchmark.py.
Can also be run standalone for a quick sanity check.
"""

import time
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Memory estimate
# Each vector: DIM * 4 bytes (float32) + M * 2 neighbour links * 4 bytes each
# ──────────────────────────────────────────────────────────────────────────────
def _mem_mb(n: int, dim: int, m: int) -> float:
    return n * (dim * 4 + m * 2 * 4) / 1024 ** 2


def run(
    data: np.ndarray,
    queries: np.ndarray,
    k: int,
    metric: str,
    gt: np.ndarray,
    *,
    m: int = 16,
    ef_construction: int = 200,
    ef_search: int = 50,
) -> dict:
    """
    Build an HNSW index on *data* and search *queries*.

    Parameters
    ----------
    data            : (N, D) float32 corpus
    queries         : (Q, D) float32 query vectors
    k               : neighbours to retrieve
    metric          : "l2" or "ip"
    gt              : (Q, k) int32 ground-truth neighbour indices
    m               : HNSW M (bi-directional links per node)
    ef_construction : build-time ef
    ef_search       : query-time ef

    Returns
    -------
    dict with keys: label, build_s, query_s, qps, recall, mem_mb
    """
    try:
        import hnswlib
    except ImportError:
        return {"label": "HNSW", "error": "hnswlib not installed — pip install hnswlib"}

    n, dim = data.shape
    space = metric  # hnswlib uses "l2" / "ip" directly

    idx = hnswlib.Index(space=space, dim=dim)
    idx.init_index(max_elements=n, ef_construction=ef_construction, M=m)

    # ── Build ──────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    idx.add_items(data)
    build_s = time.perf_counter() - t0

    # ── Search ─────────────────────────────────────────────────────────────────
    idx.set_ef(ef_search)
    t0 = time.perf_counter()
    labels, _ = idx.knn_query(queries, k=k)
    query_s = time.perf_counter() - t0

    labels = np.array(labels, dtype=np.int32)

    return {
        "label":   "HNSW",
        "build_s": build_s,
        "query_s": query_s,
        "qps":     len(queries) / query_s,
        "recall":  _recall_at_k(labels, gt),
        "mem_mb":  _mem_mb(n, dim, m),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Shared recall helper (also imported by benchmark.py)
# ──────────────────────────────────────────────────────────────────────────────
def _recall_at_k(retrieved: np.ndarray, ground_truth: np.ndarray) -> float:
    nq, k = ground_truth.shape
    hits = sum(
        len(set(retrieved[i, :k].tolist()) & set(ground_truth[i, :k].tolist()))
        for i in range(nq)
    )
    return hits / (nq * k)


# ──────────────────────────────────────────────────────────────────────────────
# Standalone quick-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng  = np.random.default_rng(42)
    data = np.ascontiguousarray(rng.standard_normal((5_000, 128)), dtype=np.float32)
    q    = np.ascontiguousarray(rng.standard_normal((100,  128)), dtype=np.float32)

    # Trivial brute-force ground truth for standalone test
    d  = np.sum((q[:, None] - data[None]) ** 2, axis=-1)
    gt = np.argsort(d, axis=1)[:, :10].astype(np.int32)

    r = run(data, q, k=10, metric="l2", gt=gt)
    print(r)
