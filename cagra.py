"""
cagra.py — cuVS CAGRA (GPU graph-based ANN) vector index wrapper.

CAGRA builds a navigable small-world graph entirely on GPU, achieving very
high QPS at excellent recall — particularly suited for large-scale, latency-
sensitive serving workloads.

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
    graph_degree: int = 64,
    itopk_size: int = 64,
) -> dict:
    """
    Build a cuVS CAGRA index on *data* and search *queries*.

    Parameters
    ----------
    data         : (N, D) float32 corpus
    queries      : (Q, D) float32 query vectors
    k            : neighbours to retrieve
    metric       : "l2" or "ip"
    gt           : (Q, k) int32 ground-truth neighbour indices
    graph_degree : out-degree of each node in the CAGRA graph
                   (higher → better recall, larger index, slower build)
    itopk_size   : internal top-k candidate list during beam search
                   (higher → better recall, lower QPS)

    Returns
    -------
    dict with keys: label, build_s, query_s, qps, recall, mem_mb
    """
    try:
        import cupy as cp
        from cuvs.neighbors import cagra
        from cuvs.common import Resources
    except ImportError:
        return {
            "label": "CAGRA (cuVS)",
            "error": "cuvs / cupy not installed — requires CUDA GPU",
        }

    res        = Resources()
    metric_str = _METRIC_MAP.get(metric, "sqeuclidean")
    params     = cagra.IndexParams(graph_degree=graph_degree, metric=metric_str)

    d_data    = cp.asarray(data)
    d_queries = cp.asarray(queries)

    # ── Build ──────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    index = cagra.build(params, d_data, resources=res)
    cp.cuda.Stream.null.synchronize()
    build_s = time.perf_counter() - t0

    # ── Search ─────────────────────────────────────────────────────────────────
    sp = cagra.SearchParams(itopk_size=itopk_size)
    t0 = time.perf_counter()
    _, labels = cagra.search(sp, index, d_queries, k, resources=res)
    cp.cuda.Stream.null.synchronize()
    query_s = time.perf_counter() - t0

    labels_np = cp.asnumpy(labels).astype(np.int32)

    # Memory: graph edges (uint32) + raw vectors
    n, dim = data.shape
    mem_mb = n * graph_degree * 4 / 1024 ** 2  # graph adjacency (uint32 per edge)

    return {
        "label":   "CAGRA (cuVS)",
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
