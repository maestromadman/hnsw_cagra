"""
ivf_pq.py — cuVS IVF-PQ (GPU, compressed) vector index wrapper.

IVF-PQ quantises each sub-vector to `pq_bits` bits, giving a much smaller
memory footprint than IVF-Flat at the cost of some recall.

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
    pq_dim: int = 0,      # 0 = auto (dim // 2)
    pq_bits: int = 8,
) -> dict:
    """
    Build a cuVS IVF-PQ index on *data* and search *queries*.

    Parameters
    ----------
    data     : (N, D) float32 corpus
    queries  : (Q, D) float32 query vectors
    k        : neighbours to retrieve
    metric   : "l2" or "ip"
    gt       : (Q, k) int32 ground-truth neighbour indices
    n_lists  : number of IVF clusters
    n_probes : clusters probed at query time
    pq_dim   : number of PQ sub-quantisers (0 = dim // 2)
    pq_bits  : bits per sub-quantiser code (4 or 8)

    Returns
    -------
    dict with keys: label, build_s, query_s, qps, recall, mem_mb
    """
    try:
        import cupy as cp
        from cuvs.neighbors import ivf_pq
        from cuvs.common import Resources
    except ImportError:
        return {
            "label": "IVF-PQ (cuVS)",
            "error": "cuvs / cupy not installed — requires CUDA GPU",
        }

    res        = Resources()
    n, dim     = data.shape
    actual_pq  = pq_dim if pq_dim > 0 else max(1, dim // 2)
    metric_str = _METRIC_MAP.get(metric, "sqeuclidean")

    params = ivf_pq.IndexParams(
        n_lists=n_lists,
        pq_dim=actual_pq,
        pq_bits=pq_bits,
        metric=metric_str,
    )

    d_data    = cp.asarray(data)
    d_queries = cp.asarray(queries)

    # ── Build ──────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    index = ivf_pq.build(params, d_data, handle=res)
    cp.cuda.Stream.null.synchronize()
    build_s = time.perf_counter() - t0

    # ── Search ─────────────────────────────────────────────────────────────────
    sp = ivf_pq.SearchParams(n_probes=n_probes)
    t0 = time.perf_counter()
    _, labels = ivf_pq.search(sp, index, d_queries, k, handle=res)
    cp.cuda.Stream.null.synchronize()
    query_s = time.perf_counter() - t0

    labels_np = cp.asnumpy(labels).astype(np.int32)

    # Memory: PQ codes — each vector is compressed to pq_dim * pq_bits bits
    mem_mb = n * actual_pq * pq_bits / 8 / 1024 ** 2

    return {
        "label":   "IVF-PQ (cuVS)",
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
