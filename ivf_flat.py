"""
ivf_flat.py — cuVS IVF-Flat (GPU) vector index wrapper.

Exposes a single `run()` function consumed by benchmark.py.
Can also be run standalone for a quick sanity check.

Requires: cupy, cuvs  (CUDA GPU)
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
    *,
    n_lists: int = 128,
    n_probes: int = 20,
) -> dict:
    """
    Build a cuVS IVF-Flat index on *data* and search *queries*.

    Parameters
    ----------
    data     : (N, D) float32 corpus
    queries  : (Q, D) float32 query vectors
    k        : neighbours to retrieve
    metric   : "l2" or "ip"
    gt       : (Q, k) int32 ground-truth neighbour indices
    n_lists  : number of IVF clusters (Voronoi cells)
    n_probes : clusters probed at query time (accuracy vs speed)

    Returns
    -------
    dict with keys: label, build_s, query_s, qps, recall, mem_mb
    """
    try:
        import cupy as cp
        from cuvs.neighbors import ivf_flat
        from cuvs.common import Resources
    except ImportError:
        return {
            "label": "IVF-Flat (cuVS)",
            "error": "cuvs / cupy not installed — requires CUDA GPU",
        }

    res        = Resources()
    metric_str = _METRIC_MAP.get(metric, "sqeuclidean")
    params     = ivf_flat.IndexParams(n_lists=n_lists, metric=metric_str)

    d_data    = cp.asarray(data)
    d_queries = cp.asarray(queries)

    # ── Build ──────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    index = ivf_flat.build(params, d_data, resources=res)
    cp.cuda.Stream.null.synchronize()
    build_s = time.perf_counter() - t0

    # ── Search ─────────────────────────────────────────────────────────────────
    sp = ivf_flat.SearchParams(n_probes=n_probes)
    t0 = time.perf_counter()
    _, labels = ivf_flat.search(sp, index, d_queries, k, resources=res)
    cp.cuda.Stream.null.synchronize()
    query_s = time.perf_counter() - t0

    labels_np = cp.asnumpy(labels).astype(np.int32)

    # Memory: raw float32 vectors (IVF-Flat stores full vectors)
    n, dim = data.shape
    mem_mb = n * dim * 4 / 1024 ** 2

    return {
        "label":   "IVF-Flat (cuVS)",
        "build_s": build_s,
        "query_s": query_s,
        "qps":     len(queries) / query_s,
        "recall":  _recall_at_k(labels_np, gt),
        "mem_mb":  mem_mb,
    }


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

    d  = np.sum((q[:, None] - data[None]) ** 2, axis=-1)
    gt = np.argsort(d, axis=1)[:, :10].astype(np.int32)

    r = run(data, q, k=10, metric="l2", gt=gt)
    print(r)
