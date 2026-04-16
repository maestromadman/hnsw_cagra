# Vector Search Benchmark: HNSW vs cuVS GPU Methods

A structured benchmarking framework comparing five vector search algorithms across
multiple real-world scenarios, built to evaluate where GPU-accelerated search
(NVIDIA cuVS) meaningfully outperforms CPU-based alternatives (hnswlib).

---

## Motivation

Vector search is the backbone of modern AI applications such as RAG, semantic search, recommendation engines, and multimodal
pipelines: they all depend on quickly and accurately locating the nearest neighbours of an embedding vector
across a large corpus.

The dominant open-source CPU solution is **hnswlib** (HNSW algorithm), widely
used in production today. NVIDIA's **cuVS** library offers GPU-accelerated
alternatives (IVF-Flat, IVF-PQ, CAGRA, Brute-Force) that promise higher
throughput at scale. The question this project answers:

> **At what scale, dimensionality, and query pattern does it make sense for a
> customer to move from CPU-based HNSW to GPU-accelerated cuVS and which
> cuVS algorithm should they choose?**

---

## Algorithms Compared

| Algorithm | Type | Hardware | Key Trade-off |
|---|---|---|---|
| **HNSW** (hnswlib) | Graph-based ANN | CPU | High recall, fast single queries, slow build at scale |
| **IVF-Flat** (cuVS) | Inverted index, exact within cluster | GPU | Good recall, moderate throughput |
| **IVF-PQ** (cuVS) | Inverted index + product quantization | GPU | Lowest memory footprint, some recall loss |
| **CAGRA** (cuVS) | GPU-optimized graph-based ANN | GPU | Highest throughput at scale |
| **Brute-Force** (cuVS) | Exhaustive exact search | GPU | Perfect recall, O(N) — baseline only |

> **Note on Brute-Force:** It serves as the ground-truth reference and an
> "exact search" baseline. Its recall is always 1.0 by definition. It is
> competitive only at small N — at scale it collapses because it has no index
> to avoid scanning every vector.

---

## Benchmark Design

All five algorithms run on **identical data** per configuration:

- Randomly generated float32 vectors (seeded for reproducibility)
- Same corpus, same query set, same k, same distance metric
- Ground truth computed via GPU brute-force
- Recall measured as Recall@k against that ground truth

**Metrics captured per algorithm:**

| Metric | What it measures |
|---|---|
| **Build time (s)** | Cost to construct the index — matters for ingestion / re-indexing |
| **QPS** | Queries per second — throughput for serving |
| **Recall@k** | Fraction of true top-k neighbours returned — search quality |
| **Est. memory (MB)** | Index size on disk/GPU — infrastructure cost |

**Three charts per run** frame results around hypothetical customer scenarios:

1. **Index Build Time + Memory** — *"How expensive is it to index my data?"*
2. **Queries per Second** — *"Can it handle my traffic?"*
3. **Recall@k vs QPS scatter** — *"Where does each method sit on the quality-speed curve?"*

---

## Experimental Runs

Seven configurations test different dimensions of the problem:

| Run | n | dim | n_queries | k | Metric | Customer Scenario |
|---|---|---|---|---|---|---|
| 1 | 10,000 | 128 | 500 | 10 | L2 | Baseline — small corpus, all methods competitive |
| 2 | 1,000,000 | 128 | 500 | 10 | L2 | Large-scale corpus — where GPU pull-ahead begins |
| 3 | 100,000 | 1,536 | 500 | 10 | L2 | Real embedding dimensions (OpenAI text-embedding-3) |
| 4 | 100,000 | 128 | 1 | 10 | L2 | Real-time single query — GPU launch overhead matters |
| 5 | 100,000 | 128 | 10,000 | 10 | L2 | Batch throughput — GPU parallelism shines |
| 6 | 100,000 | 128 | 500 | 500 | L2 | Two-stage re-ranking pipeline |
| 7 | 100,000 | 128 | 500 | 10 | IP | Cosine similarity (most embedding models) |

---

## Key Results & "Customer" Takeaways

### Run 1 — Baseline (n=10k)
At small scale, brute-force GPU dominates and all methods are fast. This is
intentiona and establishes that **at small N, no index is needed**. HNSW
build time is already 25× slower than any cuVS method.

**Takeaway:** Small corpora don't need cuVS. Value comes with scale.

---

### Run 2 — GPU at Scale (n=1M)
This is a core result. HNSW build time grows to **~5 minutes** at 1M vectors.
CAGRA builds in **~40 seconds** and reaches **~109k QPS** — roughly 8× HNSW's
throughput. Brute-force QPS collapses from 583k (at 10k) to ~10k, confirming
the trivial result that exhaustive search does not scale.

**Takeaway:** For customers with >100k vectors and throughput requirements,
CAGRA is the clear recommendation. The build time advantage alone can change
a customer's re-indexing cadence from hours to minutes.

---

### Run 3 — High Dimensions (dim=1,536, OpenAI embedding size)
At real-world embedding dimensions, memory becomes the constraint.
IVF-Flat and Brute-Force each need ~586 MB just for raw vectors.
**IVF-PQ** compresses to **73 MB** (8× reduction) while still returning
useful candidates for a downstream re-ranker.

**Takeaway:** For customers using large embedding models (OpenAI, Cohere, etc.), 
IVF-PQ is the practical choice when GPU VRAM is limited. 
Accept slightly lower recall in exchange for fitting a much larger
corpus on-device.

---

### Run 4 — Single Query / Real-Time Latency (n_queries=1)
GPU launch overhead dominates at batch size 1. **HNSW wins on QPS** here
because CPU graph traversal has near-zero overhead per query. cuVS GPU
methods pay a fixed dispatch cost regardless of batch size.

**Takeaway:** For real-time applications with single-query, low-latency
requirements (e.g., autocomplete, live recommendations), HNSW remains
competitive. cuVS is optimised for throughput, not single-query latency.
A hybrid architecture (HNSW for live traffic, cuVS for batch re-ranking)
is worth recommending to customers.

---

### Run 5 — Batch Throughput (n_queries=10k)
With large query batches, GPU parallelism dominates. **CAGRA reaches 687k
QPS** — over 30× HNSW's 22k. IVF-Flat and IVF-PQ also pull ahead
significantly. This models a nightly batch job, an offline embedding
pipeline, or a high-traffic serving cluster.

**Takeaway:** Any customer running batch inference, offline re-indexing, or
sustained high-RPS serving should be on cuVS. The throughput gap at batch
scale is the strongest argument for GPU adoption.

---

### Run 6 — Re-ranking Pipeline (k=500)
Two-stage retrieval (retrieve 500 candidates → re-rank with a cross-encoder)
is standard in production RAG. At k=500, approximate methods show recall
around 0.33–0.70. HNSW recall holds up better than IVF methods at high k.
IVF-Flat and IVF-PQ compensate with dramatically higher QPS.

**Takeaway:** For re-ranking pipelines, the first-stage retriever doesn't
need perfect recall — a cross-encoder will re-score anyway. IVF-PQ at 99k
QPS with 0.33 recall@500 is a reasonable first stage that costs far less
memory than IVF-Flat at similar throughput.

---

### Run 7 — Inner Product / Cosine Similarity
Most production embedding models (OpenAI, Cohere, Mistral) are trained with
cosine similarity, not L2 distance. Normalising vectors and using inner
product is the correct setting for these workloads. Results mirror the L2
story — the relative rankings are consistent, confirming the benchmark
generalises across distance metrics.

**Takeaway:** cuVS supports both L2 and inner product natively. Customers
don't need to change their embedding pipeline to adopt it.

---

## How to Run

**Install dependencies:**
```bash
pip install hnswlib numpy matplotlib
pip install cuvs cupy-cuda12x   # requires CUDA GPU
```

**Run all 7 configurations (saves one PNG per run):**
```bash
python benchmark.py
```

**Run a single custom configuration:**
```python
# In benchmark.py, set:
RUN_ALL = False

SINGLE_RUN = dict(
    label     = "my_test",
    n_vectors = 500_000,
    dim       = 256,
    n_queries = 1000,
    k         = 10,
    metric    = "l2",
    desc      = "My custom run",
)
```

**Run a single algorithm in isolation:**
```bash
python hnsw.py          # CPU HNSW
python ivf_flat.py      # cuVS IVF-Flat
python ivf_pq.py        # cuVS IVF-PQ
python cagra.py         # cuVS CAGRA
python brute_force_knn.py   # cuVS Brute-Force (GPU) or NumPy (CPU fallback)
```

---

## Project Structure

```
benchmark.py        — Orchestrator: runs all configs, generates charts
hnsw.py             — hnswlib CPU HNSW
ivf_flat.py         — cuVS IVF-Flat GPU
ivf_pq.py           — cuVS IVF-PQ GPU 
cagra.py            — cuVS CAGRA GPU 
brute_force_knn.py  — cuVS Brute-Force GPU (NumPy CPU fallback)
results/            — Saved benchmark PNGs
```

---

## Algorithm Selection Guide

| Customer situation | Recommended algorithm |
|---|---|
| Small corpus (<100k), any hardware | HNSW or Brute-Force |
| Large corpus (>500k), high throughput | **CAGRA** |
| GPU VRAM constrained, large embeddings | **IVF-PQ** |
| Single real-time queries, low latency | **HNSW** |
| Batch workloads, offline pipelines | **CAGRA** or **IVF-Flat** |
| Two-stage re-ranking pipeline | **IVF-PQ** (first stage) |
| Cosine similarity workloads | Any — all support inner product |

---

## Environment

- **GPU:** NVIDIA L4 (tested on GCP VM with CUDA 12.4)
- **cuVS:** RAPIDS cuVS (cuvs-env)
- **Python:** 3.10
- **Key packages:** `cuvs`, `cupy`, `hnswlib`, `numpy`, `matplotlib`
