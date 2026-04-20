"""
benchmark_2.py — Parametric cuVS ANN benchmark.

Compares IVF-Flat, IVF-PQ, and CAGRA on an NVIDIA L4 GPU (24 GB VRAM).
Customer scenario: mid-size e-commerce semantic product search using
sentence-transformer embeddings (dim=768).

Runs 4 experiments, each varying one parameter while holding others fixed:
  Exp 1 — n_vectors  : [100K, 500K, 1M, 2M, 5M]
  Exp 2 — dim        : [128, 256, 384, 512, 768, 1024]
  Exp 3 — n_queries  : [100, 500, 1K, 5K, 10K]
  Exp 4 — k          : [1, 5, 10, 50, 100]

Output: exp_{param}.png per experiment + a summary table to stdout.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Reproducibility ────────────────────────────────────────────────────────────
np.random.seed(42)
SEED = 42

# ── Control values (held fixed when not being varied) ─────────────────────────
CONTROLS = {
    "n_vectors": 1_000_000,
    "dim":       768,
    "n_queries": 1_000,
    "k":         10,
}



N_LISTS      = 256   # sqrt(65536); smaller lists → more vectors/cluster → better locality
N_PROBES     = 64   # now probes 25% of clusters (64/256) — much better recall on random data
PQ_BITS      = 8
GRAPH_DEGREE = 64
ITOPK_SIZE   = 128  # wider beam search for CAGRA; 64 was too narrow in 768-dim random space

INDEX_TYPES  = ["IVF-Flat", "IVF-PQ", "CAGRA"]
INDEX_COLORS = {
    "IVF-Flat": "tab:blue",
    "IVF-PQ":   "tab:red",
    "CAGRA":    "tab:green",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _f32(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


def _free_gpu():
    """Release cupy memory pool blocks back to the GPU allocator."""
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


_VRAM_SAFETY = 0.80  # use at most 80% of free VRAM per data point

def _check_vram(n_vectors: int, dim: int, n_queries: int):
    """
    Raise RuntimeError if the estimated peak VRAM for this data point exceeds
    the safety budget.  Estimation: 3× raw corpus bytes (data + IVF-Flat index
    copy + GT distance scratch) + queries.
    """
    import cupy as cp
    free_bytes, _total = cp.cuda.runtime.memGetInfo()
    budget = free_bytes * _VRAM_SAFETY
    # IVF-Flat worst case: raw data + index copy + GT scratch + cupy pool slack
    corpus_bytes  = n_vectors * dim * 4
    queries_bytes = n_queries * dim * 4
    estimated     = 4 * corpus_bytes + queries_bytes
    if estimated > budget:
        gb = lambda b: b / 1024**3
        raise RuntimeError(
            f"Estimated peak VRAM {gb(estimated):.1f} GB exceeds "
            f"safety budget {gb(budget):.1f} GB "
            f"(free {gb(free_bytes):.1f} GB) — skipping"
        )


def _compute_ground_truth(vectors: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Exact k-NN ground truth via cuVS GPU brute-force (GPU only)."""
    print("    [GT] computing ground truth...", end=" ", flush=True)
    t0 = time.perf_counter()
    import cupy as cp
    from cuvs.neighbors import brute_force
    from cuvs.common import Resources

    res    = Resources()
    d_data = cp.asarray(vectors)
    d_q    = cp.asarray(queries)
    bf_idx = brute_force.build(d_data, metric="sqeuclidean", resources=res)
    _, lbl = brute_force.search(bf_idx, d_q, k, resources=res)
    cp.cuda.Stream.null.synchronize()
    gt = cp.asnumpy(lbl).astype(np.int32)
    del bf_idx, d_data, d_q, res
    _free_gpu()
    print(f"GPU ({time.perf_counter() - t0:.2f}s)")
    return gt


# ── Core API ───────────────────────────────────────────────────────────────────

def build_index(index_type: str, vectors: np.ndarray, **kwargs) -> dict:
    """
    Build a cuVS index from float32 *vectors* and return an opaque handle.

    Parameters
    ----------
    index_type : "IVF-Flat" | "IVF-PQ" | "CAGRA"
    vectors    : (N, D) float32 corpus

    Returns
    -------
    Handle dict consumed by query_index().  Caller must del it when done
    and call _free_gpu() to release GPU memory.
    """
    import cupy as cp
    from cuvs.common import Resources

    _n, dim = vectors.shape
    res    = Resources()
    d_vecs = cp.asarray(vectors)

    if index_type == "IVF-Flat":
        from cuvs.neighbors import ivf_flat as _mod
        n_lists  = kwargs.get("n_lists",  N_LISTS)
        n_probes = kwargs.get("n_probes", N_PROBES)
        params   = _mod.IndexParams(n_lists=n_lists, metric="sqeuclidean")
        idx      = _mod.build(params, d_vecs, resources=res)
        cp.cuda.Stream.null.synchronize()
        del d_vecs
        sp = _mod.SearchParams(n_probes=n_probes)
        return {"type": "IVF-Flat", "index": idx, "sp": sp, "res": res, "_mod": _mod}

    elif index_type == "IVF-PQ":
        from cuvs.neighbors import ivf_pq as _mod
        n_lists  = kwargs.get("n_lists",  N_LISTS)
        n_probes = kwargs.get("n_probes", N_PROBES)
        pq_bits  = kwargs.get("pq_bits",  PQ_BITS)
        # pq_dim = dim // 8: ~8× compression; always ≤ dim for dim ≥ 8
        pq_dim   = kwargs.get("pq_dim",   max(1, dim // 8))
        params   = _mod.IndexParams(
            n_lists=n_lists, pq_dim=pq_dim, pq_bits=pq_bits, metric="sqeuclidean"
        )
        idx = _mod.build(params, d_vecs, resources=res)
        cp.cuda.Stream.null.synchronize()
        del d_vecs
        sp = _mod.SearchParams(n_probes=n_probes)
        return {"type": "IVF-PQ", "index": idx, "sp": sp, "res": res, "_mod": _mod}

    elif index_type == "CAGRA":
        from cuvs.neighbors import cagra as _mod
        graph_degree = kwargs.get("graph_degree", GRAPH_DEGREE)
        itopk_size   = kwargs.get("itopk_size",   ITOPK_SIZE)
        params       = _mod.IndexParams(graph_degree=graph_degree, metric="sqeuclidean")
        idx          = _mod.build(params, d_vecs, resources=res)
        cp.cuda.Stream.null.synchronize()
        del d_vecs
        sp = _mod.SearchParams(itopk_size=itopk_size)
        return {"type": "CAGRA", "index": idx, "sp": sp, "res": res, "_mod": _mod}

    else:
        del d_vecs
        raise ValueError(f"Unknown index_type: '{index_type}'")


def query_index(handle: dict, queries: np.ndarray, k: int):
    """
    Search *handle* for the *k* nearest neighbours of each query.

    Returns
    -------
    (indices, distances) — numpy arrays of shape (Q, k), int32 and float32.
    """
    import cupy as cp

    idx_type  = handle["type"]
    index     = handle["index"]
    sp        = handle["sp"]
    res       = handle["res"]
    _mod      = handle["_mod"]
    d_queries = cp.asarray(queries)

    if idx_type == "IVF-Flat":
        dists, labels = _mod.search(sp, index, d_queries, k, resources=res)
    elif idx_type == "IVF-PQ":
        dists, labels = _mod.search(sp, index, d_queries, k, resources=res)
    elif idx_type == "CAGRA":
        dists, labels = _mod.search(sp, index, d_queries, k, resources=res)
    else:
        raise ValueError(f"Unknown index type in handle: '{idx_type}'")

    cp.cuda.Stream.null.synchronize()
    del d_queries
    return cp.asnumpy(labels).astype(np.int32), cp.asnumpy(dists)


def compute_recall(approx_indices: np.ndarray, true_indices: np.ndarray) -> float:
    """Intersection-based Recall@k averaged over all queries."""
    nq, k = true_indices.shape
    hits = sum(
        len(set(approx_indices[i, :k].tolist()) & set(true_indices[i, :k].tolist()))
        for i in range(nq)
    )
    return hits / (nq * k)


# ── Experiment runner ──────────────────────────────────────────────────────────

def run_experiment(vary_param: str, values: list, controls: dict,
                   exp_num: int = 0, n_exps: int = 4) -> dict:
    """
    Vary *vary_param* across *values*, holding all other params at *controls*.

    For each (index_type, param_value) pair:
      - Build the index
      - Run the query phase, measure QPS
      - Compute Recall@k vs brute-force ground truth
      - Catch OOM / any exception; log and continue

    Returns
    -------
    dict:
        vary_param  : str
        values      : list
        controls    : dict
        results     : {index_type: [{"value", "qps", "recall"} | {"value", "error"}]}
    """
    results = {idx: [] for idx in INDEX_TYPES}

    for v in values:
        n_vectors = v if vary_param == "n_vectors" else controls["n_vectors"]
        dim       = v if vary_param == "dim"       else controls["dim"]
        n_queries = v if vary_param == "n_queries" else controls["n_queries"]
        k         = v if vary_param == "k"         else controls["k"]

        # Fresh generator each data point for reproducibility regardless of loop order
        # VRAM pre-check — skip before allocating anything if it won't fit
        try:
            _check_vram(n_vectors, dim, n_queries)
        except RuntimeError as e:
            print(f"  [SKIP] {vary_param}={v}: {e}")
            for idx_type in INDEX_TYPES:
                results[idx_type].append({"value": v, "qps": None, "recall": None, "error": str(e)})
            continue

        rng     = np.random.default_rng(SEED)
        vectors = _f32(rng.standard_normal((n_vectors, dim)))
        queries = _f32(rng.standard_normal((n_queries, dim)))

        # Ground truth — skip the whole data point on failure
        try:
            gt = _compute_ground_truth(vectors, queries, k)
        except Exception as e:
            msg = f"GT failed: {e}"
            print(f"  [GT FAIL] {vary_param}={v}: {msg} — skipping all indexes")
            for idx_type in INDEX_TYPES:
                results[idx_type].append({"value": v, "qps": None, "recall": None, "error": msg})
            _free_gpu()
            continue

        for idx_type in INDEX_TYPES:
            try:
                # CAGRA: itopk_size must be >= k (constraint documented in cuVS)
                kw = {}
                if idx_type == "CAGRA":
                    kw["itopk_size"] = max(ITOPK_SIZE, k)

                handle = build_index(idx_type, vectors, **kw)

                t0          = time.perf_counter()
                approx, _   = query_index(handle, queries, k)
                query_s     = time.perf_counter() - t0

                recall = compute_recall(approx, gt)
                qps    = n_queries / query_s

                print(
                    f"  Experiment {exp_num}/{n_exps}: varying {vary_param} | "
                    f"{idx_type} | {vary_param}={v} -> "
                    f"QPS: {qps:,.0f}, Recall: {recall:.4f}"
                )
                results[idx_type].append({"value": v, "qps": qps, "recall": recall})

                del handle, approx
                _free_gpu()

            except Exception as e:
                low = str(e).lower()
                is_oom = any(w in low for w in ("out of memory", "oom", "cudaerroroutofmemory", "alloc"))
                tag = "OOM" if is_oom else "ERROR"
                print(
                    f"  [{tag}] Experiment {exp_num}/{n_exps}: varying {vary_param} | "
                    f"{idx_type} | {vary_param}={v}: {e}"
                )
                results[idx_type].append({"value": v, "qps": None, "recall": None, "error": str(e)})
                _free_gpu()

    return {
        "vary_param": vary_param,
        "values":     values,
        "controls":   controls,
        "results":    results,
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_experiment(exp_results: dict, vary_param: str):
    """
    Save one PNG figure for *exp_results*.

    - X-axis: varied parameter
    - Y-axis: QPS
    - 3 lines: IVF-Flat (blue), IVF-PQ (red), CAGRA (green)
    - Each marker annotated with Recall@k value (placed just above)
    - White background, gridlines, no top/right spines
    """
    values   = exp_results["values"]
    results  = exp_results["results"]
    controls = exp_results["controls"]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for idx_type in INDEX_TYPES:
        color = INDEX_COLORS[idx_type]
        rows  = results[idx_type]
        xs    = [r["value"]  for r in rows if r.get("qps") is not None]
        ys    = [r["qps"]    for r in rows if r.get("qps") is not None]
        rs    = [r["recall"] for r in rows if r.get("qps") is not None]

        if not xs:
            ax.plot([], [], color=color, label=f"{idx_type} (no data)")
            continue

        ax.plot(xs, ys, marker="o", color=color, label=idx_type,
                linewidth=2, markersize=8, zorder=3)

        for x, y, r in zip(xs, ys, rs):
            ax.annotate(
                f"{r:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8.5,
                color=color,
                fontweight="bold",
            )

    param_label = vary_param.replace("_", " ").title()
    ctrl_parts  = [
        f"{p}={v:,}" if isinstance(v, int) else f"{p}={v}"
        for p, v in controls.items()
        if p != vary_param
    ]
    ctrl_desc = "  |  ".join(ctrl_parts)

    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel("QPS (queries per second)", fontsize=12)
    ax.set_title(
        f"QPS vs {param_label}  —  cuVS ANN Indexes on NVIDIA L4\n"
        f"E-commerce semantic product search  ·  {ctrl_desc}",
        fontsize=11,
        fontweight="bold",
        pad=14,
    )

    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.45, color="#aaaaaa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    )

    if vary_param == "n_vectors":
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda v, _: f"{int(v) // 1_000_000}M" if v >= 1e6 else f"{int(v) // 1_000}K"
            )
        )

    plt.tight_layout()
    fname = f"exp_{vary_param}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Figure saved -> {fname}")
    plt.close(fig)


# ── Summary table ──────────────────────────────────────────────────────────────

def _print_summary(all_results: list):
    col_w = [22, 12, 14, 12, 14, 10]
    header = (
        f"{'Experiment':<{col_w[0]}}"
        f"{'Index':<{col_w[1]}}"
        f"{'Parameter':<{col_w[2]}}"
        f"{'Value':<{col_w[3]}}"
        f"{'QPS':>{col_w[4]}}"
        f"{'Recall@k':>{col_w[5]}}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for exp in all_results:
        vp = exp["vary_param"]
        for idx_type in INDEX_TYPES:
            for row in exp["results"][idx_type]:
                v      = row["value"]
                qps    = row.get("qps")
                recall = row.get("recall")
                exp_lbl = f"vary_{vp}"
                if qps is not None:
                    print(
                        f"{exp_lbl:<{col_w[0]}}"
                        f"{idx_type:<{col_w[1]}}"
                        f"{vp:<{col_w[2]}}"
                        f"{str(v):<{col_w[3]}}"
                        f"{qps:>{col_w[4]},.0f}"
                        f"{recall:>{col_w[5]}.4f}"
                    )
                else:
                    err = (row.get("error") or "unknown")[:col_w[5]]
                    print(
                        f"{exp_lbl:<{col_w[0]}}"
                        f"{idx_type:<{col_w[1]}}"
                        f"{vp:<{col_w[2]}}"
                        f"{str(v):<{col_w[3]}}"
                        f"{'FAILED':>{col_w[4]}}"
                        f"{err:>{col_w[5]}}"
                    )

    print(sep)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    experiments = [
        ("n_vectors", [100_000, 250_000, 500_000, 750_000, 1_000_000]),
        ("dim",       [128, 256, 384, 512, 768]),
        ("n_queries", [100, 500, 1_000, 5_000, 10_000]),
        ("k",         [1, 5, 10, 50, 100]),
    ]

    all_results = []
    for i, (param, values) in enumerate(experiments, 1):
        ctrl_display = {p: v for p, v in CONTROLS.items() if p != param}
        print(f"\n{'=' * 64}")
        print(f"Experiment {i}/4: varying {param}")
        print(f"  Controls: {ctrl_display}")
        print(f"{'=' * 64}")

        exp_results = run_experiment(
            param, values, CONTROLS, exp_num=i, n_exps=4
        )
        plot_experiment(exp_results, param)
        all_results.append(exp_results)

    _print_summary(all_results)
    print("\nAll 4 experiments complete.")


if __name__ == "__main__":
    main()
