"""
compare_2.py — cuVS IVF-flat vs CAGRA at scale on NVIDIA L4
Usage: python compare_2.py

Produces results/comparison_2.png with 4 plots:
  1. Recall vs latency tradeoff
  2. Throughput (QPS) vs recall
  3. Index build time
  4. Latency required to hit target recall thresholds
"""

import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ------------------------------------------------------------------
# Config — 1M vectors, 128 dims
# ------------------------------------------------------------------

N         = 1_000_000
D         = 128
K         = 10
SEED      = 42
N_QUERIES = 10_000

rng     = np.random.default_rng(SEED)
dataset = rng.standard_normal((N, D)).astype(np.float32)
queries = rng.standard_normal((N_QUERIES, D)).astype(np.float32)

print(f"Dataset : {N:,} vectors × {D} dims")
print(f"Queries : {N_QUERIES:,}   k = {K}")

# ------------------------------------------------------------------
# GPU imports
# ------------------------------------------------------------------

try:
    from cuvs.neighbors import cagra, ivf_flat, brute_force
    from pylibraft.common import DeviceResources
    import cupy as cp
except ImportError:
    print("\ncuVS not available — activate the cuvs-env conda environment first.")
    sys.exit(1)


print("\nUploading data to GPU…")
dataset_gpu = cp.asarray(dataset)
queries_gpu = cp.asarray(queries)

# ------------------------------------------------------------------
# Ground truth — GPU brute force
# ------------------------------------------------------------------

print("Computing ground truth (GPU brute force)…")
t0 = time.perf_counter()
bf_index = brute_force.build(dataset_gpu, metric="sqeuclidean")
_, gt_neighbors_gpu = brute_force.search(bf_index, queries_gpu, K)
cp.cuda.Stream.null.synchronize()
gt_neighbors_cpu = cp.asnumpy(gt_neighbors_gpu)
gt = [set(gt_neighbors_cpu[i]) for i in range(N_QUERIES)]
print(f"Ground truth done in {time.perf_counter() - t0:.2f}s")

# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _recall(neighbors_cpu):
    return float(np.mean([
        len(set(neighbors_cpu[i]) & gt[i]) / K
        for i in range(N_QUERIES)
    ]))

def _lat_at_recall(recalls, latencies, target):
    """Linear interpolation: latency needed to achieve `target` recall."""
    pairs = sorted(zip(recalls, latencies))
    for i in range(len(pairs) - 1):
        r0, l0 = pairs[i]
        r1, l1 = pairs[i + 1]
        if r0 <= target <= r1 and r1 > r0:
            frac = (target - r0) / (r1 - r0)
            return l0 + frac * (l1 - l0)
    if pairs[-1][0] >= target:
        return pairs[-1][1]
    return None  # not achievable

# ------------------------------------------------------------------
# IVF-flat benchmark
# ------------------------------------------------------------------

print("\n--- cuVS IVF-flat ---")

N_LISTS      = 1024          # ≈ sqrt(1M); standard recommendation
nprobe_sweep = [1, 2, 5, 10, 20, 50, 100, 200, 400]

ivf_recalls     = []
ivf_latencies   = []
ivf_throughputs = []

bparams_ivf = ivf_flat.IndexParams(
    metric="sqeuclidean",
    n_lists=N_LISTS,
)

print(f"Building index (n_lists={N_LISTS})…")
t0 = time.perf_counter()
ivf_index = ivf_flat.build(bparams_ivf, dataset_gpu)
cp.cuda.Stream.null.synchronize()
ivf_build_time = time.perf_counter() - t0
print(f"Build time : {ivf_build_time:.2f}s")

print(f"\n{'n_probes':>10}  {'recall@'+str(K):>12}  {'lat (ms)':>10}  {'QPS':>10}")
print("-" * 50)

for nprobe in nprobe_sweep:
    sparams = ivf_flat.SearchParams(n_probes=nprobe)

    # warmup
    ivf_flat.search(sparams, ivf_index, queries_gpu[:64], K)
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    _, nbrs_gpu = ivf_flat.search(sparams, ivf_index, queries_gpu, K)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0

    nbrs   = cp.asnumpy(nbrs_gpu)
    rc     = _recall(nbrs)
    lat_ms = elapsed / N_QUERIES * 1000
    qps    = N_QUERIES / elapsed

    ivf_recalls.append(rc)
    ivf_latencies.append(lat_ms)
    ivf_throughputs.append(qps)
    print(f"{nprobe:>10}  {rc:>12.4f}  {lat_ms:>9.3f}ms  {qps:>10,.0f}")

# ------------------------------------------------------------------
# CAGRA benchmark
# ------------------------------------------------------------------

print("\n--- CAGRA ---")

itopk_sweep = [16, 32, 64, 128, 256, 512]

cagra_recalls     = []
cagra_latencies   = []
cagra_throughputs = []

bparams_cagra = cagra.IndexParams(
    metric="sqeuclidean",
    graph_degree=64,
    intermediate_graph_degree=128,
)

print("Building index…")
t0 = time.perf_counter()
cagra_index = cagra.build(bparams_cagra, dataset_gpu)
cp.cuda.Stream.null.synchronize()
cagra_build_time = time.perf_counter() - t0
print(f"Build time : {cagra_build_time:.2f}s")

print(f"\n{'itopk':>10}  {'recall@'+str(K):>12}  {'lat (ms)':>10}  {'QPS':>10}")
print("-" * 50)

for itopk in itopk_sweep:
    sparams = cagra.SearchParams(itopk_size=itopk)

    # warmup
    cagra.search(sparams, cagra_index, queries_gpu[:64], K)
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    _, nbrs_gpu = cagra.search(sparams, cagra_index, queries_gpu, K)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0

    nbrs   = cp.asnumpy(nbrs_gpu)
    rc     = _recall(nbrs)
    lat_ms = elapsed / N_QUERIES * 1000
    qps    = N_QUERIES / elapsed

    cagra_recalls.append(rc)
    cagra_latencies.append(lat_ms)
    cagra_throughputs.append(qps)
    print(f"{itopk:>10}  {rc:>12.4f}  {lat_ms:>9.3f}ms  {qps:>10,.0f}")

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------

print("\nGenerating comparison_2.png…")

IVF_COLOR   = "#378ADD"
CAGRA_COLOR = "#1D9E75"

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

# ── Plot 1: Recall vs latency ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(ivf_latencies,   ivf_recalls,   "o-", color=IVF_COLOR,   lw=2, ms=6, label="IVF-flat")
ax1.plot(cagra_latencies, cagra_recalls, "s-", color=CAGRA_COLOR, lw=2, ms=6, label="CAGRA")
ax1.set_xlabel("Latency per query (ms)")
ax1.set_ylabel(f"Recall@{K}")
ax1.set_title("Recall vs latency tradeoff")
ax1.set_ylim(0, 1.05)
ax1.axhline(1.0, color="#ccc", lw=0.8, linestyle="--")
ax1.legend()
ax1.grid(True, alpha=0.2)

# ── Plot 2: QPS vs recall ──────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(ivf_recalls,   ivf_throughputs,   "o-", color=IVF_COLOR,   lw=2, ms=6, label="IVF-flat")
ax2.plot(cagra_recalls, cagra_throughputs, "s-", color=CAGRA_COLOR, lw=2, ms=6, label="CAGRA")
ax2.set_xlabel(f"Recall@{K}")
ax2.set_ylabel("Throughput (queries / sec)")
ax2.set_title("Throughput vs recall")
ax2.legend()
ax2.grid(True, alpha=0.2)

# ── Plot 3: Build time ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
bar_labels = [f"IVF-flat\n(n_lists={N_LISTS})", "CAGRA\n(graph°=64)"]
bar_times  = [ivf_build_time, cagra_build_time]
bar_colors = [IVF_COLOR, CAGRA_COLOR]
bars = ax3.bar(bar_labels, bar_times, color=bar_colors, width=0.4, alpha=0.85)
for bar, val in zip(bars, bar_times):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 1.02,
        f"{val:.2f}s",
        ha="center", va="bottom", fontsize=11, fontweight="500",
    )
ax3.set_ylabel("Build time (s)")
ax3.set_title(f"Index build time  (N={N:,}, D={D})")
ax3.grid(True, alpha=0.2, axis="y")

# ── Plot 4: Latency to hit target recall ───────────────────────────
ax4   = fig.add_subplot(gs[1, 1])
targets = [0.80, 0.90, 0.95, 0.99]
x       = np.arange(len(targets))
width   = 0.32

ivf_lats_at  = [_lat_at_recall(ivf_recalls,   ivf_latencies,   t) for t in targets]
cagra_lats_at = [_lat_at_recall(cagra_recalls, cagra_latencies, t) for t in targets]

ivf_bars   = ax4.bar(x - width/2, ivf_lats_at,   width, color=IVF_COLOR,   alpha=0.85, label="IVF-flat")
cagra_bars = ax4.bar(x + width/2, cagra_lats_at, width, color=CAGRA_COLOR, alpha=0.85, label="CAGRA")

for bar, val in zip(list(ivf_bars) + list(cagra_bars),
                    ivf_lats_at + cagra_lats_at):
    if val is not None:
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=8,
        )
    else:
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            0.5,
            "N/A",
            ha="center", va="bottom", fontsize=8, color="gray",
        )

ax4.set_xticks(x)
ax4.set_xticklabels([f"{int(t*100)}% recall" for t in targets])
ax4.set_ylabel("Latency per query (ms)")
ax4.set_title("Latency required to hit target recall")
ax4.legend()
ax4.grid(True, alpha=0.2, axis="y")

plt.suptitle(
    f"cuVS IVF-flat vs CAGRA  —  N={N:,}, D={D}, k={K}  —  NVIDIA L4",
    fontsize=13, fontweight="600", y=1.02,
)

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/comparison_2.png", dpi=150, bbox_inches="tight")
print("Saved to results/comparison_2.png")

# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------

print("\n" + "=" * 64)
print(f"{'SUMMARY  (N=1M, D=128, k=10, L4 GPU)':^64}")
print("=" * 64)
print(f"{'Metric':<34} {'IVF-flat':>14} {'CAGRA':>14}")
print("-" * 64)
print(f"{'Build time (s)':<34} {ivf_build_time:>14.2f} {cagra_build_time:>14.2f}")
print(f"{'Max recall@'+str(K):<34} {max(ivf_recalls):>14.4f} {max(cagra_recalls):>14.4f}")
print(f"{'Min latency (ms)':<34} {min(ivf_latencies):>14.3f} {min(cagra_latencies):>14.3f}")
print(f"{'Peak throughput (QPS)':<34} {max(ivf_throughputs):>14,.0f} {max(cagra_throughputs):>14,.0f}")
for t in [0.80, 0.90, 0.95, 0.99]:
    ivf_l  = _lat_at_recall(ivf_recalls,   ivf_latencies,   t)
    cag_l  = _lat_at_recall(cagra_recalls, cagra_latencies, t)
    ivf_s  = f"{ivf_l:.3f}ms"  if ivf_l  is not None else "N/A"
    cag_s  = f"{cag_l:.3f}ms"  if cag_l  is not None else "N/A"
    print(f"{'Latency @ '+str(int(t*100))+'% recall':<34} {ivf_s:>14} {cag_s:>14}")
print("=" * 64)
