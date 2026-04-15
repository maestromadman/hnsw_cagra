"""
benchmark.py — Vector search benchmark orchestrator.

Runs all five methods under identical conditions and produces three
scenario-framed matplotlib charts:

  Chart 1 — "Data Ingestion"      : index build time + memory footprint
  Chart 2 — "High-Traffic Serving": queries per second
  Chart 3 — "Quality vs Speed"    : Recall@k vs QPS scatter (bubble = memory)

Edit the CONFIG block to change parameters, then run:
    python benchmark.py

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
# CONFIG — edit these to change the benchmark
# ──────────────────────────────────────────────────────────────────────────────

N_VECTORS  = 10_000   # total vectors in the index
DIM        = 128      # vector dimensionality
N_QUERIES  = 500      # number of query vectors
K          = 10       # k for k-NN
METRIC     = "l2"     # "l2" or "ip" (inner product / cosine on normalised vecs)
SEED       = 42

# ── HNSW ──────────────────────────────────────────────────────────────────────
HNSW_M               = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH       = 50

# ── IVF shared ────────────────────────────────────────────────────────────────
N_LISTS  = 128
N_PROBES = 20

# ── IVF-PQ ────────────────────────────────────────────────────────────────────
PQ_DIM  = 0   # 0 → auto (DIM // 2)
PQ_BITS = 8

# ── CAGRA ─────────────────────────────────────────────────────────────────────
CAGRA_GRAPH_DEGREE = 64
CAGRA_ITOPK        = 64

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette (one colour per algorithm — consistent across all charts)
# ──────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "HNSW":                   "#4C72B0",  # blue
    "IVF-Flat (cuVS)":        "#DD8452",  # orange
    "IVF-PQ (cuVS)":          "#55A868",  # green
    "CAGRA (cuVS)":           "#C44E52",  # red
    "Brute-Force (cuVS GPU)": "#8172B2",  # purple
    "Brute-Force (CPU)":      "#8172B2",  # same purple (CPU fallback label)
}
DEFAULT_COLOR = "#999999"


def _color(label):
    for key, c in PALETTE.items():
        if key in label:
            return c
    return DEFAULT_COLOR


# ──────────────────────────────────────────────────────────────────────────────
# Ground-truth helper
# ──────────────────────────────────────────────────────────────────────────────
def _f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def compute_ground_truth(data, queries, k, metric):
    """Exact brute-force ground truth computed on CPU with NumPy."""
    print("  [GT] computing brute-force ground truth ...", end=" ", flush=True)
    t0 = time.perf_counter()
    if metric == "l2":
        chunk, rows = 200, []
        for s in range(0, len(queries), chunk):
            d = np.sum((queries[s : s + chunk, None] - data[None]) ** 2, axis=-1)
            idx = np.argpartition(d, k, axis=1)[:, :k]
            for i in range(len(idx)):
                idx[i] = idx[i][np.argsort(d[i, idx[i]])]
            rows.append(idx)
        gt = np.vstack(rows).astype(np.int32)
    else:
        d  = -(queries @ data.T)
        gt = np.argsort(d, axis=1)[:, :k].astype(np.int32)
    print(f"done ({time.perf_counter() - t0:.2f}s)")
    return gt


# ──────────────────────────────────────────────────────────────────────────────
# Plot — 3 scenario-framed subplots
# ──────────────────────────────────────────────────────────────────────────────
def plot(results, n_vectors, dim, k, metric):
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
        f"Vector Search Benchmark  —  n={n_vectors:,}  dim={dim}  k={k}  metric={metric}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    gs = fig.add_gridspec(1, 3, wspace=0.40)

    # ── Chart 1: Build time + memory  (Data Ingestion scenario) ──────────────
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

    # Overlay memory as a secondary-axis dashed line with diamond markers
    ax1r = ax1.twinx()
    ax1r.plot(x, mem_mb, color="#333333", marker="D", markersize=5,
              linewidth=1.2, linestyle="--", label="Est. index size (MB)")
    ax1r.set_ylabel("Est. index size (MB)", fontsize=8.5)
    ax1r.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax1r.spines[["top"]].set_visible(False)
    ax1r.legend(fontsize=8, loc="upper right")

    # ── Chart 2: QPS  (High-traffic real-time serving scenario) ──────────────
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

    # ── Chart 3: Recall@k vs QPS  (Quality-critical semantic search) ─────────
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

    # Shared legend at the bottom
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
    out = f"benchmark_n{n_vectors}_dim{dim}_k{k}_{metric}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved \u2192 {out}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Runner table — one entry per algorithm
# ──────────────────────────────────────────────────────────────────────────────
RUNNERS = [
    (
        "HNSW",
        lambda data, queries, k, metric, gt: hnsw.run(
            data, queries, k, metric, gt,
            m=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION, ef_search=HNSW_EF_SEARCH,
        ),
    ),
    (
        "IVF-Flat (cuVS)",
        lambda data, queries, k, metric, gt: ivf_flat.run(
            data, queries, k, metric, gt,
            n_lists=N_LISTS, n_probes=N_PROBES,
        ),
    ),
    (
        "IVF-PQ (cuVS)",
        lambda data, queries, k, metric, gt: ivf_pq.run(
            data, queries, k, metric, gt,
            n_lists=N_LISTS, n_probes=N_PROBES, pq_dim=PQ_DIM, pq_bits=PQ_BITS,
        ),
    ),
    (
        "CAGRA (cuVS)",
        lambda data, queries, k, metric, gt: cagra.run(
            data, queries, k, metric, gt,
            graph_degree=CAGRA_GRAPH_DEGREE, itopk_size=CAGRA_ITOPK,
        ),
    ),
    (
        "Brute-Force",
        lambda data, queries, k, metric, gt: brute_force_knn.run(
            data, queries, k, metric, gt,
        ),
    ),
]


def main():
    print("=" * 64)
    print("  Vector Search Benchmark")
    print(f"  n={N_VECTORS:,}  dim={DIM}  nq={N_QUERIES}  k={K}  metric={METRIC}")
    print("=" * 64)

    rng     = np.random.default_rng(SEED)
    data    = _f32(rng.standard_normal((N_VECTORS, DIM)))
    queries = _f32(rng.standard_normal((N_QUERIES,  DIM)))

    if METRIC == "ip":
        data    /= np.linalg.norm(data,    axis=1, keepdims=True)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    gt = compute_ground_truth(data, queries, K, METRIC)

    results = []
    for name, runner in RUNNERS:
        print(f"\n  >> {name} ...", flush=True)
        r = runner(data, queries, K, METRIC, gt)
        results.append(r)
        if "error" in r:
            print(f"     SKIP: {r['error']}")
        else:
            print(
                f"     build={r['build_s']:.3f}s  "
                f"query={r['query_s']:.4f}s  "
                f"QPS={r['qps']:,.0f}  "
                f"recall@{K}={r['recall']:.4f}  "
                f"mem={r['mem_mb']:.1f}MB"
            )

    plot(results, N_VECTORS, DIM, K, METRIC)
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
