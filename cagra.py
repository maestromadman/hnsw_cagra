"""
cagra.py — CAGRA benchmark using NVIDIA cuVS
Builds a CAGRA index on the same dataset as hnsw.py and measures:
  - Build time
  - Recall@10 across a sweep of itopk_size (CAGRA's ef_search equivalent)
  - Query latency per itopk_size value
 
Saves results to results/cagra_results.npz for compare.py to load.
"""
 
import numpy as np
import time
import os
 
from cuvs.neighbors import cagra
from pylibraft.common import DeviceResources
import cupy as cp
 
# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
 
N = 50000
D = 128
K = 10
SEED = 42
N_QUERIES = 500
 
rng = np.random.default_rng(SEED)
dataset = rng.standard_normal((N, D)).astype(np.float32)
queries = rng.standard_normal((N_QUERIES, D)).astype(np.float32)
 
print(f"Dataset: {N} vectors, {D} dimensions")
print(f"Queries: {N_QUERIES}")
 
# ------------------------------------------------------------------
# Brute force ground truth (on CPU)
# ------------------------------------------------------------------
 
print("\nComputing ground truth...")
t0 = time.perf_counter()
gt = []
for q in queries:
    dists = np.sum((dataset - q) ** 2, axis=1)
    gt.append(set(np.argsort(dists)[:K]))
gt_time = time.perf_counter() - t0
print(f"Ground truth computed in {gt_time:.2f}s")
 
# ------------------------------------------------------------------
# Build CAGRA index
# ------------------------------------------------------------------
 
print("\nBuilding CAGRA index...")
 
handle = DeviceResources()
dataset_gpu = cp.asarray(dataset)
 
build_params = cagra.IndexParams(
    metric="sqeuclidean",
    graph_degree=64,
    intermediate_graph_degree=128,
)
 
t0 = time.perf_counter()
index = cagra.build(build_params, dataset_gpu, handle=handle)
handle.sync()
build_time = time.perf_counter() - t0
 
print(f"CAGRA build time: {build_time:.3f}s")
 
# ------------------------------------------------------------------
# Recall vs itopk_size sweep
# itopk_size is CAGRA's beam width — equivalent to ef_search in HNSW
# ------------------------------------------------------------------
 
itopk_values = [16, 32, 64, 128, 256, 512]
recalls = []
latencies = []
 
queries_gpu = cp.asarray(queries)
 
print(f"\n{'itopk':>8}  {'recall@'+str(K):>12}  {'latency(ms)':>12}")
print("-" * 36)
 
for itopk in itopk_values:
    search_params = cagra.SearchParams(itopk_size=itopk)
 
    # warmup
    _ = cagra.search(search_params, index, queries_gpu[:10], K, handle=handle)
    handle.sync()
 
    t0 = time.perf_counter()
    distances_gpu, neighbors_gpu = cagra.search(
        search_params, index, queries_gpu, K, handle=handle
    )
    handle.sync()
    elapsed = time.perf_counter() - t0
 
    neighbors_cpu = cp.asnumpy(neighbors_gpu)
 
    recall = float(np.mean([
        len(set(neighbors_cpu[i]) & gt[i]) / K
        for i in range(N_QUERIES)
    ]))
    lat_ms = elapsed / N_QUERIES * 1000
 
    recalls.append(recall)
    latencies.append(lat_ms)
    print(f"{itopk:>8}  {recall:>12.4f}  {lat_ms:>11.3f}ms")
 
# ------------------------------------------------------------------
# Save results
# ------------------------------------------------------------------
 
os.makedirs("results", exist_ok=True)
np.savez(
    "results/cagra_results.npz",
    itopk_values=np.array(itopk_values),
    recalls=np.array(recalls),
    latencies=np.array(latencies),
    build_time=np.array(build_time),
    N=np.array(N),
    D=np.array(D),
    K=np.array(K),
)
print("\nResults saved to results/cagra_results.npz")