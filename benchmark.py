"""
benchmark.py — Vector search benchmark orchestrator.

Runs all five methods under identical conditions and produces three
scenario-framed matplotlib charts per run configuration:

  Chart 1 — "Data Ingestion"      : index build time + memory footprint
  Chart 2 — "High-Traffic Serving": queries per second
  Chart 3 — "Quality vs Speed"    : Recall@k vs QPS scatter (bubble = memory)

To run all preset configurations automatically:
    python benchmark.py

To run a single custom configuration, edit the SINGLE_RUN block at the
bottom and set RUN_ALL = False.

Dependencies:
    pip install hnswlib numpy matplotlib
    (cuVS methods require a CUDA GPU + pip install cuvs cupy-cudaXXX)
"""

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# Import algorithm modules
import hnsw
import ivf_flat
import ivf_pq
import cagra
import brute_force_knn

# ──────────────────────────────────────────────────────────────────────────────
# RUN MODE
#   RUN_ALL = True  → loop through all EXPERIMENT_RUNS below (one PNG each)
#   RUN_ALL = False → run only SINGLE_RUN config below
# ──────────────────────────────────────────────────────────────────────────────
RUN_ALL = True

# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENT RUNS  (only used when RUN_ALL = True)
# Each dict needs: n_vectors, dim, n_queries, k, metric, label
# ──────────────────────────────────────────────────────────────────────────────
EXPERIMENT_RUNS = [
    dict(
        label     = "run1_baseline",
        n_vectors = 10_000,
        dim       = 128,
        n_queries = 500,
        k         = 10,
        metric    = "l2",
        desc      = "Baseline",
    ),
    dict(
        label     = "run2_scale",
        n_vectors = 1_000_000,
        dim       = 128,
        n_queries = 500,
        k         = 10,
        metric    = "l2",
        desc      = "GPU at scale",
    ),
    dict(
        label     = "run3_highdim",
        n_vectors = 100_000,
        dim       = 1536,
        n_queries = 500,
        k         = 10,
        metric    = "l2",
        desc      = "Real embedding dims (OpenAI text-embedding-3)",
    ),
    dict(
        label     = "run4_single_query",
        n_vectors = 100_000,
        dim       = 128,
        n_queries = 1,
        k         = 10,
        metric    = "l2",
        desc      = "Real-time single query (GPU overhead vs HNSW)",
    ),
    dict(
        label     = "run5_batch",
        n_vectors = 100_000,
        dim       = 128,
        n_queries = 10_000,
        k         = 10,
        metric    = "l2",
        desc      = "Batch throughput",
    ),
    dict(
        label     = "run6_rerank",
        n_vectors = 100_000,
        dim       = 128,
        n_queries = 500,
        k         = 500,
        metric    = "l2",
        desc      = "Re-ranking pipeline (k=500)",
    ),
    dict(
        label     = "run7_innerproduct",
        n_vectors = 100_000,
        dim       = 128,
        n_queries = 500,
        k         = 10,
        metric    = "ip",
        desc      = "Cosine similarity / inner product",
    ),
]

# ──────────────────────────────────────────────────────────────────────────────
# SINGLE RUN CONFIG  (only used when RUN_ALL = False)
# ──────────────────────────────────────────────────────────────────────────────
SINGLE_RUN = dict(
    label     = "custom",
    n_vectors = 10_000,
    dim       = 128,
    n_queries = 500,
    k         = 10,
    metric    = "l2",
    desc      = "Custom run",
)

# ──────────────────────────────────────────────────────────────────────────────
# ALGORITHM TUNING PARAMS  (shared across all runs)
# ──────────────────────────────────────────────────────────────────────────────
SEED                 = 42
HNSW_M               = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH       = 50
N_LISTS              = 128
N_PROBES             = 20
PQ_DIM               = 0    # 0 → auto (dim // 2)
PQ_BITS              = 8
CAGRA_GRAPH_DEGREE   = 64
CAGRA_ITOPK          = 64

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette — consistent across all runs and charts
# ──────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "HNSW":                   "#4C72B0",
    "IVF-Flat (cuVS)":        "#DD8452",
    "IVF-PQ (cuVS)":          "#55A868",
    "CAGRA (cuVS)":           "#C44E52",
    "Brute-Force (cuVS GPU)": "#8172B2",
    "Brute-Force (CPU)":      "#8172B2",
}
DEFAULT_COLOR = "#999999"


def _color(label):
    for key, c in PALETTE.items():
        if key in label:
            return c
    return DEFAULT_COLOR


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def compute_ground_truth(data, queries, k, metric):
    """
    Exact brute-force ground truth.
    Tries cuVS GPU first (fast, memory-efficient at scale).
    Falls back to chunked NumPy CPU (1 query at a time to avoid OOM).
    """
    print("  [GT] computing brute-force ground truth ...", end=" ", flush=True)
    t0 = time.perf_counter()

    try:
        import cupy as cp
        from cuvs.neighbors import brute_force
        from cuvs.common import Resources

        res        = Resources()
        metric_str = "sqeuclidean" if metric == "l2" else "inner_product"
        d_data     = cp.asarray(data)
        d_queries  = cp.asarray(queries)
        index      = brute_force.build(d_data, metric=metric_str, resources=res)
        _, labels  = brute_force.search(index, d_queries, k, resources=res)
        cp.cuda.Stream.null.synchronize()
        gt = cp.asnumpy(labels).astype(np.int32)
        del index, d_data, d_queries, res
        print(f"done via GPU ({time.perf_counter() - t0:.2f}s)")

    except ImportError:
        # CPU fallback: one query at a time to keep memory bounded at any N
        rows = []
        for i in range(len(queries)):
            if metric == "l2":
                d   = np.sum((queries[i] - data) ** 2, axis=-1)
                idx = np.argpartition(d, k)[:k]
                idx = idx[np.argsort(d[idx])]
            else:
                d   = -(queries[i] @ data.T)
                idx = np.argsort(d)[:k]
            rows.append(idx)
        gt = np.array(rows, dtype=np.int32)
        print(f"done via CPU ({time.perf_counter() - t0:.2f}s)")

    return gt


# ──────────────────────────────────────────────────────────────────────────────
# Plot — 3 scenario-framed subplots
# ──────────────────────────────────────────────────────────────────────────────
def plot(results, cfg):
    n_vectors = cfg["n_vectors"]
    dim       = cfg["dim"]
    k         = cfg["k"]
    metric    = cfg["metric"]
    label     = cfg["label"]
    desc      = cfg.get("desc", "")

    valid  = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    if failed:
        print("\n  Skipped (missing deps):")
        for r in failed:
            print(f"    - {r['label']}: {r['error']}")

    if not valid:
        print("  No valid results — nothing to plot.")
        return

    labels   = [r["label"]   for r in valid]
    build_s  = [r["build_s"] for r in valid]
    qps_v    = [r["qps"]     for r in valid]
    mem_mb   = [r["mem_mb"]  for r in valid]
    colors   = [_color(lbl)  for lbl in labels]

    x      = np.arange(len(valid))
    bar_kw = dict(edgecolor="white", linewidth=0.9, width=0.55)

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        f"Vector Search Benchmark  —  {desc}\n"
        f"n={n_vectors:,}  dim={dim}  k={k}  metric={metric}",
        fontsize=13, fontweight="bold", y=1.03,
    )
    gs = fig.add_gridspec(1, 3, wspace=0.40)

    # ── Chart 1: Build time + memory ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    bars = ax1.bar(x, build_s, color=colors, **bar_kw)
    ax1.set_title(
        "Index Build Time  (seconds)\n"
        r"$\it{Scenario:\ data\ ingestion\ /\ re\text{-}indexing}$",
        fontsize=10,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
    ax1.set_ylabel("Build time (s)")
    ax1.bar_label(bars, fmt="%.3fs", padding=3, fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    ax1r = ax1.twinx()
    ax1r.plot(x, mem_mb, color="#333333", marker="D", markersize=5,
              linewidth=1.2, linestyle="--", label="Est. index size (MB)")
    ax1r.set_ylabel("Est. index size (MB)", fontsize=8.5)
    ax1r.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax1r.spines[["top"]].set_visible(False)
    ax1r.legend(fontsize=8, loc="upper right")

    # ── Chart 2: QPS ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    bars = ax2.bar(x, qps_v, color=colors, **bar_kw)
    ax2.set_title(
        "Queries per Second  (QPS)\n"
        r"$\it{Scenario:\ high\text{-}traffic\ real\text{-}time\ serving}$",
        fontsize=10,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
    ax2.set_ylabel("QPS  (higher = better)")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax2.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    # ── Chart 3: Recall vs QPS scatter ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])

    max_mem   = max(mem_mb) or 1.0
    min_sz, max_sz = 80, 600
    bubble_sz = [min_sz + (m / max_mem) * (max_sz - min_sz) for m in mem_mb]

    for r, c, bsz in zip(valid, colors, bubble_sz):
        ax3.scatter(
            r["recall"], r["qps"],
            s=bsz, color=c, alpha=0.82,
            edgecolors="white", linewidths=0.9, zorder=3,
        )
        ax3.annotate(
            r["label"],
            (r["recall"], r["qps"]),
            textcoords="offset points", xytext=(8, 4),
            fontsize=8, color=c,
        )

    ax3.set_title(
        f"Recall@{k}  vs  QPS  (bubble \u221d memory)\n"
        r"$\it{Scenario:\ quality\text{-}critical\ semantic\ search}$",
        fontsize=10,
    )
    ax3.set_xlabel(f"Recall@{k}  (higher = better)")
    ax3.set_ylabel("QPS  (higher = better)")
    ax3.set_xlim(-0.02, 1.08)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax3.grid(True, linestyle="--", alpha=0.35)
    ax3.spines[["top", "right"]].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=_color(lbl), label=lbl) for lbl in labels
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=len(valid),
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )

    plt.tight_layout()
    out = f"{label}_n{n_vectors}_dim{dim}_k{k}_{metric}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved \u2192 {out}")
    plt.close(fig)   # free memory between runs


# ──────────────────────────────────────────────────────────────────────────────
# Core: run one configuration
# ──────────────────────────────────────────────────────────────────────────────
def run_config(cfg):
    n_vectors = cfg["n_vectors"]
    dim       = cfg["dim"]
    n_queries = cfg["n_queries"]
    k         = cfg["k"]
    metric    = cfg["metric"]
    desc      = cfg.get("desc", "")

    print("\n" + "=" * 64)
    print(f"  {desc}")
    print(f"  n={n_vectors:,}  dim={dim}  nq={n_queries}  k={k}  metric={metric}")
    print("=" * 64)

    rng     = np.random.default_rng(SEED)
    data    = _f32(rng.standard_normal((n_vectors, dim)))
    queries = _f32(rng.standard_normal((n_queries,  dim)))

    if metric == "ip":
        data    /= np.linalg.norm(data,    axis=1, keepdims=True)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    gt = compute_ground_truth(data, queries, k, metric)

    runners = [
        ("HNSW",
         lambda d, q, k_, m, g: hnsw.run(
             d, q, k_, m, g,
             m=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION, ef_search=HNSW_EF_SEARCH,
         )),
        ("IVF-Flat (cuVS)",
         lambda d, q, k_, m, g: ivf_flat.run(
             d, q, k_, m, g,
             n_lists=N_LISTS, n_probes=N_PROBES,
         )),
        ("IVF-PQ (cuVS)",
         lambda d, q, k_, m, g: ivf_pq.run(
             d, q, k_, m, g,
             n_lists=N_LISTS, n_probes=N_PROBES, pq_dim=PQ_DIM, pq_bits=PQ_BITS,
         )),
        ("CAGRA (cuVS)",
         lambda d, q, k_, m, g: cagra.run(
             d, q, k_, m, g,
             graph_degree=CAGRA_GRAPH_DEGREE, itopk_size=CAGRA_ITOPK,
         )),
        ("Brute-Force",
         lambda d, q, k_, m, g: brute_force_knn.run(d, q, k_, m, g)),
    ]

    results = []
    for name, runner in runners:
        print(f"\n  >> {name} ...", flush=True)
        r = runner(data, queries, k, metric, gt)
        results.append(r)
        if "error" in r:
            print(f"     SKIP: {r['error']}")
        else:
            print(
                f"     build={r['build_s']:.3f}s  "
                f"query={r['query_s']:.4f}s  "
                f"QPS={r['qps']:,.0f}  "
                f"recall@{k}={r['recall']:.4f}  "
                f"mem={r['mem_mb']:.1f}MB"
            )

    plot(results, cfg)
    print(f"\n  Run '{desc}' complete.\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if RUN_ALL:
        print(f"Running {len(EXPERIMENT_RUNS)} configurations...\n")
        for i, cfg in enumerate(EXPERIMENT_RUNS, 1):
            print(f"[{i}/{len(EXPERIMENT_RUNS)}] {cfg['desc']}")
            run_config(cfg)
        print("\nAll runs complete.")
    else:
        run_config(SINGLE_RUN)
