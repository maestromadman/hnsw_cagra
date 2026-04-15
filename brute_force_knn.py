"""
brute_force_knn.py — cuVS brute-force KNN (GPU) vector search wrapper.

Computes exact nearest neighbours by exhaustive distance calculation on GPU.
This is the accuracy baseline — recall is always 1.0 — but it does not scale
to large corpora because search complexity is O(N·Q).

Falls back to a NumPy CPU implementation if cuVS / CuPy are unavailable.

Exposes a single `run()` function consumed by benchmark.py.
Can also be run standalone for a quick sanity check.
"""

import time
import numpy as np


_METRIC_MAP = {
    "l2": "sqeuclidean",
    "ip": "inner_product",
}


def run(
    data: np.ndarray,
    queries: np.ndarray,
    k: int,
    metric: str,
    gt: np.ndarray,
) -> dict:
    """
    Exact brute-force KNN search.

    Tries cuVS GPU first; falls back to NumPy CPU if unavailable.

    Parameters
    ----------
    data    : (N, D) float32 corpus
    queries : (Q, D) float32 query vectors
    k       : neighbours to retrieve
    metric  : "l2" or "ip"
    gt      : (Q, k) int32 ground-truth neighbour indices

    Returns
    -------
    dict with keys: label, build_s, query_s, qps, recall, mem_mb
    """
    try:
        return _run_cuvs(data, queries, k, metric, gt)
    except ImportError:
        return _run_numpy(data, queries, k, metric, gt)


# ──────────────────────────────────────────────────────────────────────────────
# cuVS GPU path
# ──────────────────────────────────────────────────────────────────────────────
def _run_cuvs(data, queries, k, metric, gt):
    import cupy as cp
    from cuvs.neighbors import brute_force
    from cuvs.common import Resources

    res        = Resources()
    metric_str = _METRIC_MAP.get(metric, "sqeuclidean")

    d_data    = cp.asarray(data)
    d_queries = cp.asarray(queries)

    # ── Build (transfer + index creation) ─────────────────────────────────────
    t0 = time.perf_counter()
    index = brute_force.build(d_data, metric=metric_str, resources=res)
    cp.cuda.Stream.null.synchronize()
    build_s = time.perf_counter() - t0

    # ── Search ─────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    _, labels = brute_force.search(index, d_queries, k, resources=res)
    cp.cuda.Stream.null.synchronize()
    query_s = time.perf_counter() - t0

    labels_np = cp.asnumpy(labels).astype(np.int32)
    n, dim    = data.shape

    return {
        "label":   "Brute-Force (cuVS GPU)",
        "build_s": build_s,
        "query_s": query_s,
        "qps":     len(queries) / query_s,
        "recall":  _recall_at_k(labels_np, gt),
        "mem_mb":  n * dim * 4 / 1024 ** 2,
    }


# ──────────────────────────────────────────────────────────────────────────────
# NumPy CPU fallback
# ──────────────────────────────────────────────────────────────────────────────
def _run_numpy(data, queries, k, metric, gt):
    n, dim = data.shape
    chunk  = 200

    t0 = time.perf_counter()
    if metric == "l2":
        rows = []
        for s in range(0, len(queries), chunk):
            d = np.sum((queries[s : s + chunk, None] - data[None]) ** 2, axis=-1)
            rows.append(np.argpartition(d, k, axis=1)[:, :k])
        labels = np.vstack(rows).astype(np.int32)
    else:
        d      = -(queries @ data.T)
        labels = np.argsort(d, axis=1)[:, :k].astype(np.int32)
    query_s = time.perf_counter() - t0

    return {
        "label":   "Brute-Force (CPU)",
        "build_s": 0.0,
        "query_s": query_s,
        "qps":     len(queries) / query_s,
        "recall":  _recall_at_k(labels, gt),
        "mem_mb":  n * dim * 4 / 1024 ** 2,
    }


def _recall_at_k(retrieved: np.ndarray, ground_truth: np.ndarray) -> float:
    nq, k = ground_truth.shape
    hits  = sum(
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

    d  = np.sum((q[:, None] - data[None]) ** 2, axis=-1)
    gt = np.argsort(d, axis=1)[:, :10].astype(np.int32)

    r = run(data, q, k=10, metric="l2", gt=gt)
    print(r)
