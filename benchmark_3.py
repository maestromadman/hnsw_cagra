"""
benchmark_3.py — Recall-QPS Pareto frontier.

Builds each index ONCE at fixed scale, then sweeps the quality knob
(n_probes for IVF indexes, itopk_size for CAGRA) to trace the full
speed-accuracy tradeoff curve for each index.

IVF-PQ is run at four compression levels (pq_dim = 48/96/192/384)
to show how the quantization ceiling rises as compression decreases.

Runtime: ~5-8 minutes.
Output:  exp_pareto.png
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark_2 import (
    _free_gpu, _free_cpu,
    _load_data, _compute_ground_truth, compute_recall,
    N_LISTS, PQ_BITS, GRAPH_DEGREE,
)

# ── Fixed parameters ──────────────────────────────────────────────────────────
N_VECTORS = 500_000
DIM       = 768
N_QUERIES = 500
K         = 10

# Quality knobs to sweep (low → high quality)
IVF_FLAT_PROBES = [1, 4, 8, 16, 32, 64, 128, 192, 256]
IVF_PQ_PROBES   = [1, 4, 8, 16, 32, 64, 128, 192, 256]
CAGRA_ITOPK     = [32, 48, 64, 96, 128, 192, 256]

# pq_dim values to compare: dim//16, dim//8, dim//4, dim//2
IVF_PQ_DIMS     = [48, 96, 192, 384]

# Red shades from light to dark for each pq_dim level
IVF_PQ_COLORS   = {
    48:  "#ffaaaa",
    96:  "#e05050",
    192: "#b00000",
    384: "#600000",
}


# ── Sweep helper (index already built; only search params vary) ───────────────

def _sweep(label, param_name, param_values, search_fn, d_queries, queries, gt):
    import cupy as cp
    results = []
    for p in param_values:
        t0 = time.perf_counter()
        labels = search_fn(p, d_queries)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0
        recall = compute_recall(cp.asnumpy(labels).astype(np.int32), gt)
        qps    = len(queries) / elapsed
        results.append({"param": p, "recall": recall, "qps": qps})
        print(f"  {label:<20} {param_name}={p:3d}  QPS={qps:9,.0f}  Recall={recall:.4f}")
    return results


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_pareto(all_results):
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    plot_order = ["IVF-Flat"] + [f"IVF-PQ (pq_dim={d})" for d in IVF_PQ_DIMS] + ["CAGRA"]
    colors = {
        "IVF-Flat": "tab:blue",
        "CAGRA":    "tab:green",
        **{f"IVF-PQ (pq_dim={d})": IVF_PQ_COLORS[d] for d in IVF_PQ_DIMS},
    }
    styles = {
        "IVF-Flat": "-",
        "CAGRA":    "-",
        **{f"IVF-PQ (pq_dim={d})": "--" for d in IVF_PQ_DIMS},
    }

    for name in plot_order:
        if name not in all_results:
            continue
        results = all_results[name]
        color   = colors[name]
        ls      = styles[name]
        xs      = [r["recall"] for r in results]
        ys      = [r["qps"]    for r in results]
        params  = [r["param"]  for r in results]

        ax.plot(xs, ys, marker="o", color=color, label=name,
                linewidth=2, markersize=7, zorder=3, linestyle=ls)

        for x, y, p in zip(xs, ys, params):
            ax.annotate(
                str(p),
                (x, y),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=7,
                color=color,
            )

    ax.set_xlabel("Recall@10", fontsize=12)
    ax.set_ylabel("QPS (queries per second)", fontsize=12)
    ax.set_title(
        "Recall-QPS Pareto Frontier  —  cuVS ANN Indexes on NVIDIA L4\n"
        "500K vectors · dim=768 · k=10 · 500 queries"
        "  |  labels = n_probes (IVF) / itopk_size (CAGRA)",
        fontsize=10, fontweight="bold", pad=14,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.45, color="#aaaaaa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    plt.tight_layout()
    plt.savefig("exp_pareto.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("\nFigure saved -> exp_pareto.png")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    vectors, queries = _load_data(N_VECTORS, N_QUERIES, DIM)

    print("Computing ground truth...")
    gt = _compute_ground_truth(vectors, queries, K)

    import cupy as cp
    from cuvs.common import Resources

    all_results = {}

    # ── IVF-Flat ──────────────────────────────────────────────────────────────
    print("\nBuilding IVF-Flat...")
    from cuvs.neighbors import ivf_flat
    res    = Resources()
    d_vecs = cp.asarray(vectors)
    idx    = ivf_flat.build(
        ivf_flat.IndexParams(n_lists=N_LISTS, metric="sqeuclidean"),
        d_vecs, resources=res,
    )
    cp.cuda.Stream.null.synchronize()
    del d_vecs
    _free_gpu()

    print("Sweeping IVF-Flat n_probes...")
    d_q = cp.asarray(queries)
    all_results["IVF-Flat"] = _sweep(
        "IVF-Flat", "n_probes", IVF_FLAT_PROBES,
        lambda p, dq: ivf_flat.search(ivf_flat.SearchParams(n_probes=p), idx, dq, K, resources=res)[1],
        d_q, queries, gt,
    )
    del d_q, idx, res
    _free_gpu()

    # ── IVF-PQ (multiple compression levels) ─────────────────────────────────
    from cuvs.neighbors import ivf_pq
    for pq_dim in IVF_PQ_DIMS:
        label = f"IVF-PQ (pq_dim={pq_dim})"
        print(f"\nBuilding {label}...")
        res    = Resources()
        d_vecs = cp.asarray(vectors)
        idx    = ivf_pq.build(
            ivf_pq.IndexParams(
                n_lists=N_LISTS, pq_dim=pq_dim, pq_bits=PQ_BITS, metric="sqeuclidean"
            ),
            d_vecs, resources=res,
        )
        cp.cuda.Stream.null.synchronize()
        del d_vecs
        _free_gpu()

        print(f"Sweeping {label} n_probes...")
        d_q = cp.asarray(queries)
        all_results[label] = _sweep(
            label, "n_probes", IVF_PQ_PROBES,
            lambda p, dq: ivf_pq.search(ivf_pq.SearchParams(n_probes=p), idx, dq, K, resources=res)[1],
            d_q, queries, gt,
        )
        del d_q, idx, res
        _free_gpu()

    # ── CAGRA ─────────────────────────────────────────────────────────────────
    print("\nBuilding CAGRA (slowest step)...")
    from cuvs.neighbors import cagra
    res    = Resources()
    d_vecs = cp.asarray(vectors)
    idx    = cagra.build(
        cagra.IndexParams(graph_degree=GRAPH_DEGREE, metric="sqeuclidean"),
        d_vecs, resources=res,
    )
    cp.cuda.Stream.null.synchronize()
    del d_vecs
    _free_gpu()

    print("Sweeping CAGRA itopk_size...")
    d_q = cp.asarray(queries)
    all_results["CAGRA"] = _sweep(
        "CAGRA", "itopk", CAGRA_ITOPK,
        lambda p, dq: cagra.search(cagra.SearchParams(itopk_size=max(p, K)), idx, dq, K, resources=res)[1],
        d_q, queries, gt,
    )
    del d_q, idx, res
    _free_gpu()

    # ── Plot & cleanup ────────────────────────────────────────────────────────
    plot_pareto(all_results)

    del vectors, queries, gt
    _free_cpu()
    _free_gpu()
    print("Done.")


if __name__ == "__main__":
    main()
