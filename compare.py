"""
compare.py — runs both HNSW and CAGRA benchmarks then plots comparison
Usage: python compare.py

Produces results/comparison.png with 3 plots:
  1. Recall vs beam width (ef_search / itopk_size)
  2. Recall vs latency tradeoff curve
  3. Build time comparison bar chart
"""

import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ------------------------------------------------------------------
# Shared dataset — both benchmarks use identical data
# ------------------------------------------------------------------

N = 10000
D = 128
K = 10
SEED = 42
N_QUERIES = 500

rng = np.random.default_rng(SEED)
dataset = rng.standard_normal((N, D)).astype(np.float32)
queries = rng.standard_normal((N_QUERIES, D)).astype(np.float32)

print(f"Dataset: {N} vectors, {D} dims, {N_QUERIES} queries, k={K}")

# ------------------------------------------------------------------
# Ground truth (brute force on CPU)
# ------------------------------------------------------------------

print("\nComputing ground truth...")
gt = []
for q in queries:
    dists = np.sum((dataset - q) ** 2, axis=1)
    gt.append(set(np.argsort(dists)[:K]))
print("Done.")

# ------------------------------------------------------------------
# HNSW benchmark
# ------------------------------------------------------------------

print("\n--- HNSW ---")
from hnsw import HNSW

ef_values = [10, 20, 50, 100, 200, 400]
hnsw_recalls = []
hnsw_latencies = []

print("Building index...")
t0 = time.perf_counter()
index = HNSW(M=16, ef_construction=200, seed=SEED)
for v in dataset:
    index.insert(v)
hnsw_build_time = time.perf_counter() - t0
print(f"Build time: {hnsw_build_time:.2f}s")

print(f"\n{'ef_search':>10}  {'recall@'+str(K):>12}  {'latency(ms)':>12}")
print("-" * 38)

for ef in ef_values:
    t0 = time.perf_counter()
    results = [
        set(nid for _, nid in index.query(q, k=K, ef_search=ef))
        for q in queries
    ]
    elapsed = time.perf_counter() - t0

    recall = float(np.mean([len(r & g) / K for r, g in zip(results, gt)]))
    lat_ms = elapsed / N_QUERIES * 1000

    hnsw_recalls.append(recall)
    hnsw_latencies.append(lat_ms)
    print(f"{ef:>10}  {recall:>12.4f}  {lat_ms:>11.3f}ms")

# ------------------------------------------------------------------
# hnswlib
# ------------------------------------------------------------------
 
print("\n--- hnswlib (production CPU) ---")
import hnswlib
 
ef_values_lib = [10, 20, 50, 100, 200, 400]
lib_recalls = []
lib_latencies = []
 
print("Building index...")
t0 = time.perf_counter()
lib_index = hnswlib.Index(space='l2', dim=D)
lib_index.init_index(max_elements=N, ef_construction=200, M=16)
lib_index.add_items(dataset)
lib_build_time = time.perf_counter() - t0
print(f"Build time: {lib_build_time:.3f}s")
 
print(f"\n{'ef_search':>10}  {'recall@'+str(K):>12}  {'latency(ms)':>12}")
print("-" * 38)
 
for ef in ef_values_lib:
    lib_index.set_ef(ef)
    t0 = time.perf_counter()
    labels, _ = lib_index.knn_query(queries, k=K)
    elapsed = time.perf_counter() - t0
    recall = float(np.mean([
        len(set(labels[i]) & gt[i]) / K
        for i in range(N_QUERIES)
    ]))
    lat_ms = elapsed / N_QUERIES * 1000
    lib_recalls.append(recall)
    lib_latencies.append(lat_ms)
    print(f"{ef:>10}  {recall:>12.4f}  {lat_ms:>11.3f}ms")
    

# ------------------------------------------------------------------
# CAGRA benchmark
# ------------------------------------------------------------------

print("\n--- CAGRA ---")

try:
    from cuvs.neighbors import cagra
    from pylibraft.common import DeviceResources
    import cupy as cp
except ImportError:
    print("cuVS not available — skipping CAGRA benchmark.")
    print("Run in cuvs-env: conda activate cuvs-env")
    sys.exit(1)

itopk_values = [16, 32, 64, 128, 256, 512]
cagra_recalls = []
cagra_latencies = []

dataset_gpu = cp.asarray(dataset)
queries_gpu = cp.asarray(queries)

build_params = cagra.IndexParams(
    metric="sqeuclidean",
    graph_degree=64,
    intermediate_graph_degree=128,
)

print("Building index...")
t0 = time.perf_counter()
cagra_index = cagra.build(build_params, dataset_gpu)
cagra_build_time = time.perf_counter() - t0
print(f"Build time: {cagra_build_time:.3f}s")

print(f"\n{'itopk':>10}  {'recall@'+str(K):>12}  {'latency(ms)':>12}")
print("-" * 38)

for itopk in itopk_values:
    search_params = cagra.SearchParams(itopk_size=itopk)

    # warmup
    _ = cagra.search(search_params, cagra_index, queries_gpu[:10], K)
    

    t0 = time.perf_counter()
    _, neighbors_gpu = cagra.search(
        search_params, cagra_index, queries_gpu, K
    )
    elapsed = time.perf_counter() - t0

    neighbors_cpu = cp.asnumpy(neighbors_gpu)
    recall = float(np.mean([
        len(set(neighbors_cpu[i]) & gt[i]) / K
        for i in range(N_QUERIES)
    ]))
    lat_ms = elapsed / N_QUERIES * 1000

    cagra_recalls.append(recall)
    cagra_latencies.append(lat_ms)
    print(f"{itopk:>10}  {recall:>12.4f}  {lat_ms:>11.3f}ms")

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
 
print("\nGenerating comparison plot...")
 
OUR_COLOR   = "#888780"
LIB_COLOR   = "#378ADD"
CAGRA_COLOR = "#1D9E75"
 
fig = plt.figure(figsize=(16, 5))
gs = gridspec.GridSpec(1, 3, figure=fig)
 
# --- Plot 1: Recall vs beam width ---
ax1 = fig.add_subplot(gs[0])
ax1.plot(ef_values,     our_recalls,   "o--", color=OUR_COLOR,   lw=2, label="Our HNSW")
ax1.plot(ef_values_lib, lib_recalls,   "o-",  color=LIB_COLOR,   lw=2, label="hnswlib")
ax1.plot(itopk_values,  cagra_recalls, "s-",  color=CAGRA_COLOR, lw=2, label="CAGRA (GPU)")
ax1.set_xlabel("Beam width (ef / itopk)")
ax1.set_ylabel(f"Recall@{K}")
ax1.set_title("Recall vs beam width")
ax1.set_ylim(0, 1.05)
ax1.axhline(1.0, color="#ccc", lw=0.8, linestyle="--")
ax1.legend()
ax1.grid(True, alpha=0.2)
 
# --- Plot 2: Recall vs latency ---
ax2 = fig.add_subplot(gs[1])
ax2.plot(our_latencies,   our_recalls,   "o--", color=OUR_COLOR,   lw=2, label="Our HNSW")
ax2.plot(lib_latencies,   lib_recalls,   "o-",  color=LIB_COLOR,   lw=2, label="hnswlib")
ax2.plot(cagra_latencies, cagra_recalls, "s-",  color=CAGRA_COLOR, lw=2, label="CAGRA (GPU)")
ax2.set_xlabel("Latency per query (ms)")
ax2.set_ylabel(f"Recall@{K}")
ax2.set_title("Recall vs latency tradeoff")
ax2.set_ylim(0, 1.05)
ax2.legend()
ax2.grid(True, alpha=0.2)
 
# --- Plot 3: Build time ---
ax3 = fig.add_subplot(gs[2])
labels_bar = ["Our HNSW\n(CPU)", "hnswlib\n(CPU)", "CAGRA\n(GPU L4)"]
times_bar  = [our_build_time, lib_build_time, cagra_build_time]
colors_bar = [OUR_COLOR, LIB_COLOR, CAGRA_COLOR]
bars = ax3.bar(labels_bar, times_bar, color=colors_bar, width=0.4, alpha=0.85)
for bar, val in zip(bars, times_bar):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{val:.2f}s",
        ha="center", va="bottom", fontsize=10, fontweight="500"
    )
ax3.set_ylabel("Build time (s)")
ax3.set_title(f"Index build time\n(N={N:,}, D={D})")
ax3.grid(True, alpha=0.2, axis="y")
 
plt.suptitle(
    f"Our HNSW vs hnswlib vs CAGRA  —  N={N:,}, D={D}, k={K}  —  NVIDIA L4",
    fontsize=12, fontweight="500", y=1.02
)
 
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/comparison.png", dpi=150, bbox_inches="tight")
print("Saved to results/comparison.png")
 
# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------
 
print("\n" + "=" * 62)
print(f"{'SUMMARY':^62}")
print("=" * 62)
print(f"{'Metric':<30} {'Our HNSW':>10} {'hnswlib':>10} {'CAGRA':>8}")
print("-" * 62)
print(f"{'Build time (s)':<30} {our_build_time:>10.2f} {lib_build_time:>10.3f} {cagra_build_time:>8.3f}")
print(f"{'Max recall@'+str(K):<30} {max(our_recalls):>10.4f} {max(lib_recalls):>10.4f} {max(cagra_recalls):>8.4f}")
print(f"{'Min latency (ms)':<30} {min(our_latencies):>10.3f} {min(lib_latencies):>10.3f} {min(cagra_latencies):>8.3f}")
print(f"{'Build speedup vs ours':<30} {'1.0x':>10} {our_build_time/lib_build_time:>9.1f}x {our_build_time/cagra_build_time:>7.1f}x")
print("=" * 62)
