import numpy as np
import heapq

class HNSW:
    def __init__(self, M=16, ef_construction=200, distance='euclidean', seed=None):
        self.M = M
        self.M0 = 2 * M                    # layer 0 gets more neighbors
        self.ef_construction = ef_construction
        self.mL = 1.0 / np.log(M)         # controls layer assignment spread

        if distance == 'euclidean':
            self.dist = HNSW.euclidean
        elif distance == 'cosine':
            self.dist = HNSW.cosine
        else:
            raise ValueError(f"Unknown distance: {distance}")

        self.rng = np.random.default_rng(seed)

        self.vectors = []        # list of np.ndarray — the actual data
        self.graphs = []         # graphs[layer][node_id] = [neighbor_ids]
        self.entry_point = None  # single entry node at the top layer
        self.max_layer = -1

    # ------------------------------------------------------------------
    # Distance functions
    # ------------------------------------------------------------------

    @staticmethod
    def euclidean(a, b):
        diff = a - b
        return float(np.dot(diff, diff))   # squared L2, no sqrt needed

    @staticmethod
    def cosine(a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 1.0
        return 1.0 - float(np.dot(a, b) / denom)

    # ------------------------------------------------------------------
    # Layer sampling
    # ------------------------------------------------------------------

    def sample_layer(self):
        # exponential distribution — most nodes land at 0,
        # probability roughly halves with each layer up
        return min(int(-np.log(self.rng.uniform()) * self.mL), 16)

    # ------------------------------------------------------------------
    # Core search primitive
    # ------------------------------------------------------------------

    def search_layer(self, query, entry_points, ef, layer):
        visited = set(entry_points)
        candidates = []   # min-heap: (dist, node_id) — what to explore next
        found = []        # max-heap: (-dist, node_id) — best ef results so far

        for ep in entry_points:
            d = self.dist(query, self.vectors[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(found, (-d, ep))

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)

            worst_found = -found[0][0]
            if c_dist > worst_found:
                break   # can't improve — stop

            for neighbor_id in self.graphs[layer].get(c_id, []):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)

                n_dist = self.dist(query, self.vectors[neighbor_id])
                worst_found = -found[0][0]

                if n_dist < worst_found or len(found) < ef:
                    heapq.heappush(candidates, (n_dist, neighbor_id))
                    heapq.heappush(found, (-n_dist, neighbor_id))
                    if len(found) > ef:
                        heapq.heappop(found)   # evict worst

        return sorted((-d, nid) for d, nid in found)

    # ------------------------------------------------------------------
    # Neighbor selection
    # ------------------------------------------------------------------

    def select_neighbors_simple(self, candidates, M):
        # just take the M closest — candidates is already sorted ascending
        return [nid for _, nid in candidates[:M]]

    def select_neighbors_heuristic(self, query, candidates, M):
        # keep only neighbors that point in genuinely new directions
        selected = []

        for dist_to_query, cid in candidates:
            if len(selected) >= M:
                break

            is_useful = all(
                dist_to_query < self.dist(self.vectors[cid], self.vectors[s])
                for s in selected
            )

            if not selected or is_useful:
                selected.append(cid)

        return selected

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def insert(self, vector, use_heuristic=True):
        node_id = len(self.vectors)
        self.vectors.append(np.array(vector, dtype=np.float32))

        node_layer = self.sample_layer()

        while len(self.graphs) <= node_layer:
            self.graphs.append({})

        # first node — no search, no edges, just register and return
        if self.entry_point is None:
            self.entry_point = node_id
            self.max_layer = node_layer
            for lc in range(node_layer + 1):
                self.graphs[lc][node_id] = []
            return node_id

        ep = [self.entry_point]

        # phase 1: fast greedy descent from top layer to node_layer+1
        # ef=1 means take only the single best at each layer — just getting close
        for lc in range(self.max_layer, node_layer, -1):
            results = self.search_layer(vector, ep, ef=1, layer=lc)
            ep = [results[0][1]]

        # phase 2: proper beam search from node_layer down to 0
        # this is where we actually find and wire up neighbors
        for lc in range(min(node_layer, self.max_layer), -1, -1):
            results = self.search_layer(vector, ep, ef=self.ef_construction, layer=lc)

            M_layer = self.M0 if lc == 0 else self.M
            if use_heuristic:
                neighbors = self.select_neighbors_heuristic(vector, results, M_layer)
            else:
                neighbors = self.select_neighbors_simple(results, M_layer)

            # connect new node to its selected neighbors
            self.graphs[lc][node_id] = neighbors

            # connect each neighbor back to new node (bidirectional edges)
            for nb in neighbors:
                if nb not in self.graphs[lc]:
                    self.graphs[lc][nb] = []
                self.graphs[lc][nb].append(node_id)

                # if neighbor now exceeds M connections, prune it back down
                if len(self.graphs[lc][nb]) > M_layer:
                    nb_candidates = [
                        (self.dist(self.vectors[nb], self.vectors[x]), x)
                        for x in self.graphs[lc][nb]
                    ]
                    if use_heuristic:
                        self.graphs[lc][nb] = self.select_neighbors_heuristic(
                            self.vectors[nb], nb_candidates, M_layer
                        )
                    else:
                        self.graphs[lc][nb] = self.select_neighbors_simple(
                            nb_candidates, M_layer
                        )

            # carry best candidates down as entry points for next layer
            ep = [nid for _, nid in results]

        # if new node was promoted above current max, it becomes the new entry point
        if node_layer > self.max_layer:
            self.max_layer = node_layer
            self.entry_point = node_id

        return node_id

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, vector, k=10, ef_search=None):
        if self.entry_point is None:
            return []

        if ef_search is None:
            ef_search = max(k, self.ef_construction // 2)
        ef_search = max(ef_search, k)

        vector = np.array(vector, dtype=np.float32)
        ep = [self.entry_point]

        # phase 1: fast descent from top layer to layer 1
        for lc in range(self.max_layer, 0, -1):
            results = self.search_layer(vector, ep, ef=1, layer=lc)
            ep = [results[0][1]]

        # phase 2: full beam search at layer 0
        results = self.search_layer(vector, ep, ef=ef_search, layer=0)

        return results[:k]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self):
        n = len(self.vectors)
        layer_counts = [len(g) for g in self.graphs]
        avg_degree = [
            round(float(np.mean([len(nb) for nb in g.values()])), 2) if g else 0
            for g in self.graphs
        ]
        return {
            "n_vectors": n,
            "max_layer": self.max_layer,
            "entry_point": self.entry_point,
            "nodes_per_layer": layer_counts,
            "avg_degree_per_layer": avg_degree,
        }
