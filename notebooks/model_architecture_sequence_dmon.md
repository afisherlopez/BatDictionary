# Model Architecture: Sequence-Level Clustering with GAT + DMoN

The goal is to cluster 4-gram sequences of bat chirps. This architecture uses a GAT encoder to create context-rich chirp embeddings, combines them into sequence embeddings, and then applies DMoN clustering at the sequence level.

Dataset: 41,774 chirps across 2,929 recordings → approximately 41,774 - 3×2,929 ≈ 32,987 overlapping 4-gram sequences.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     STAGE 1: Chirp Encoding                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Chirp features X ∈ ℝ^(n × 30)                                │
│           │                                                     │
│           ▼                                                     │
│   ┌───────────────────┐                                        │
│   │   GAT Encoder     │  (2 layers, 4 heads)                   │
│   │   + Masking       │  Self-supervised reconstruction        │
│   └───────────────────┘                                        │
│           │                                                     │
│           ▼                                                     │
│   Chirp embeddings H ∈ ℝ^(n × 32)                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STAGE 2: Sequence Construction                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   For each 4-gram (i, i+1, i+2, i+3):                          │
│                                                                 │
│   Add positional encodings: h'_j = h_j + p_j                   │
│                                                                 │
│   Concatenate: h_s = [h'_i || h'_{i+1} || h'_{i+2} || h'_{i+3}]│
│                                                                 │
│   h_s ∈ ℝ^128                                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                STAGE 3: Sequence Embedding + DMoN               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────┐                                          │
│   │  Sequence MLP   │  128 → 64 → 32                           │
│   └─────────────────┘                                          │
│           │                                                     │
│           ▼                                                     │
│   Sequence embeddings S ∈ ℝ^(N_seq × 32)                       │
│           │                                                     │
│           ├──────────────────────┐                             │
│           ▼                      ▼                             │
│   ┌──────────────┐      ┌───────────────┐                      │
│   │ Build k-NN   │      │  DMoN Cluster │                      │
│   │ Sequence     │ ──── │  Head         │                      │
│   │ Graph        │      │  (softmax)    │                      │
│   └──────────────┘      └───────────────┘                      │
│                                │                               │
│                                ▼                               │
│                    Cluster assignments C ∈ ℝ^(N_seq × k)       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Chirp Encoding (GAT with Reconstruction)

### 1.1 Preprocessing

- Reduce chirp features to 30 PCA components
- Normalize features globally (zero mean, unit variance)
- For each recording r with n_r chirps:
  - Node features: X_r ∈ ℝ^(n_r × 30)
  - Edge indices: connect temporally adjacent chirps (bidirectional)
  - Edge features: e_{i,i+1} = log(1 + Δt), normalized

### 1.2 GAT Encoder

**First Layer**

Input: x_i ∈ ℝ^30

For each of 4 attention heads (h = 1, 2, 3, 4):

1. Linear projections:
   - q_i^(h) = W_Q^(h) x_i,  k_i^(h) = W_K^(h) x_i,  v_i^(h) = W_V^(h) x_i
   - W_Q^(h), W_K^(h), W_V^(h) ∈ ℝ^(8×30)

2. Edge features: z_{ij} = φ_e(e_{ij}) ∈ ℝ^4

3. Attention:
   - ℓ_{ij}^(h) = u^(h) · LeakyReLU([q_i^(h) || k_j^(h) || z_{ij}])
   - α_{ij}^(h) = softmax_j(ℓ_{ij}^(h))

4. Aggregate: h̃_i^(h) = Σ_{j∈N(i)} α_{ij}^(h) v_j^(h)

5. Concatenate: h̃_i = SeLU([h̃_i^(1) || h̃_i^(2) || h̃_i^(3) || h̃_i^(4)]) ∈ ℝ^32

**Second Layer**

Same structure with W_Q^(h), W_K^(h), W_V^(h) ∈ ℝ^(8×32)

Output: H ∈ ℝ^(n × 32)

### 1.3 Reconstruction Head (Self-Supervision)

Decoder MLP: f : ℝ^32 → ℝ^30

- Linear(32 → 64) → ReLU → Linear(64 → 30)

For masked chirps: x̂_i = f(h_i)

**Loss:** L_recon = (1/|M|) Σ_{i∈M} ||x̂_i - x_i||_2^2

---

## Stage 2: Sequence Construction

### 2.1 Positional Encodings

Add learned positional encodings to distinguish position within the 4-gram:

- P ∈ ℝ^(4 × 32) — learnable positional embedding matrix
- p_1, p_2, p_3, p_4 are the four position embeddings

For the j-th chirp in a 4-gram (j ∈ {1,2,3,4}):

h'_j = h_j + p_j

### 2.2 Sequence Formation

For each valid 4-gram starting at chirp i (within a single recording):

h_s^(i) = [h'_i || h'_{i+1} || h'_{i+2} || h'_{i+3}] ∈ ℝ^128

**Note:** 4-grams do not cross recording boundaries.

Total sequences: N_seq = Σ_r (n_r - 3) ≈ 32,987

---

## Stage 3: Sequence Embedding and Clustering

### 3.1 Sequence MLP

Transform concatenated chirp embeddings into a unified sequence representation:

g : ℝ^128 → ℝ^32

Architecture:
- Linear(128 → 64) → SeLU → Dropout(0.2) → Linear(64 → 32)

Output: s_i = g(h_s^(i)) ∈ ℝ^32

Collect all sequence embeddings: S ∈ ℝ^(N_seq × 32)

### 3.2 Sequence Graph Construction (k-NN)

DMoN requires a graph structure to compute modularity. We construct a k-nearest-neighbor graph over sequence embeddings.

**Construction:**

1. Compute pairwise distances (or similarities) between sequence embeddings
2. For each sequence, connect to its k nearest neighbors
3. Symmetrize: if i→j or j→i exists, add edge {i,j}

**Efficient implementation:**

- Use approximate k-NN (e.g., FAISS, Annoy, or PyTorch-based) for scalability
- k_neighbors = 15-30 typically works well
- Recompute graph every few epochs (not every batch) to reduce cost

**Adjacency matrix:** A_seq ∈ ℝ^(N_seq × N_seq), sparse

**Edge weights (optional):**
- Unweighted: A_{ij} = 1 if connected
- Weighted: A_{ij} = exp(-||s_i - s_j||^2 / τ) for temperature τ

### 3.3 DMoN Clustering Head

**Cluster assignment:**

C = softmax(S · W_C + b_C)

where:
- W_C ∈ ℝ^(32 × k), b_C ∈ ℝ^k
- k = number of sequence clusters (hyperparameter)
- Apply dropout (p = 0.5) to S before projection

Output: C ∈ ℝ^(N_seq × k), where C_{ij} = P(sequence i ∈ cluster j)

### 3.4 DMoN Loss

**Modularity loss:**

L_mod = -(1/2m) Tr(C^T B C)

where:
- B = A_seq - (d d^T)/(2m) is the modularity matrix
- d_i = Σ_j A_{ij} (degree of sequence node i)
- m = (1/2) Σ_{ij} A_{ij} (total edges)

**Efficient computation:**

L_mod = -(1/2m) [Tr(C^T A_seq C) - ||d^T C||^2 / (2m)]

**Collapse regularization:**

L_collapse = (√k / N_seq) ||Σ_i C_i||_F - 1

**Combined DMoN loss:**

L_DMoN = L_mod + L_collapse

---

## Joint Training

### Loss Function

**Total loss:**

L_total = L_recon + λ · L_DMoN

where λ controls the clustering strength (start with λ = 1.0).

### Training Loop

```python
for epoch in range(num_epochs):
    
    # === Stage 1: Chirp encoding (per recording) ===
    all_chirp_embeddings = []
    total_recon_loss = 0
    
    for recording in recordings:
        X, A_chirp, edge_attr = recording.data()
        
        # Mask 10% of chirps
        mask = torch.rand(X.shape[0]) < 0.1
        X_masked = X.clone()
        X_masked[mask] = 0  # or Gaussian noise
        
        # GAT forward
        H = gat_encoder(X_masked, A_chirp, edge_attr)
        all_chirp_embeddings.append(H)
        
        # Reconstruction loss
        X_hat = decoder(H[mask])
        total_recon_loss += mse_loss(X_hat, X[mask])
    
    L_recon = total_recon_loss / len(recordings)
    
    # === Stage 2: Build sequences ===
    sequences = []
    for r, H_r in enumerate(all_chirp_embeddings):
        n_r = H_r.shape[0]
        for i in range(n_r - 3):
            # Add positional encodings
            h_seq = torch.cat([
                H_r[i] + pos_embed[0],
                H_r[i+1] + pos_embed[1],
                H_r[i+2] + pos_embed[2],
                H_r[i+3] + pos_embed[3]
            ])
            sequences.append(h_seq)
    
    H_seq = torch.stack(sequences)  # (N_seq, 128)
    
    # === Stage 3: Sequence embedding + clustering ===
    S = sequence_mlp(H_seq)  # (N_seq, 32)
    
    # Build k-NN graph (every N epochs for efficiency)
    if epoch % knn_update_freq == 0:
        A_seq = build_knn_graph(S.detach(), k=k_neighbors)
    
    # DMoN clustering
    C = softmax(dropout(S) @ W_C + b_C)
    
    # DMoN loss
    L_DMoN = modularity_loss(C, A_seq) + collapse_loss(C, k)
    
    # Total loss
    L_total = L_recon + lambda_dmon * L_DMoN
    
    # Backward pass (gradients flow through everything)
    L_total.backward()
    optimizer.step()
```

### Gradient Flow

The key insight is that gradients flow all the way back:

```
L_DMoN → C → S → sequence_mlp → H_seq → H → gat_encoder → X
```

This means the GAT learns chirp embeddings that produce good sequence clusters, not just good reconstructions.

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Chirp GAT** | | |
| Hidden dim | 32 | Output embedding size |
| Attention heads | 4 | 8 dims per head |
| GAT layers | 2 | |
| Dropout (features) | 0.1 | |
| Dropout (attention) | 0.1 | |
| Mask probability | 0.1 | For reconstruction |
| **Sequence MLP** | | |
| Architecture | 128 → 64 → 32 | With SeLU, dropout |
| Dropout | 0.2 | |
| **k-NN Graph** | | |
| k_neighbors | 15-30 | Start with 20 |
| Update frequency | Every 5 epochs | Balance cost vs freshness |
| Edge weights | Unweighted or Gaussian | Try both |
| **DMoN** | | |
| k (clusters) | 16-64 | Start with 32 |
| Dropout (pre-softmax) | 0.5 | Critical for convergence |
| λ (loss weight) | 0.1 - 1.0 | Tune based on loss scales |
| **Optimization** | | |
| Optimizer | Adam | |
| Learning rate | 10^(-3) | |
| Weight decay | 10^(-5) | |
| Epochs | 50-100 | With early stopping |

---

## Alternative: Temporal Sequence Graph

Instead of k-NN based on embedding similarity, you could define edges based on temporal/structural relationships:

**Option A: Overlapping sequences**
- Connect sequences that share chirps
- Sequence (i, i+1, i+2, i+3) shares 3 chirps with (i+1, i+2, i+3, i+4)
- Creates a chain structure within each recording

**Option B: Cross-recording similarity**
- Connect sequences across recordings if they're similar
- Encourages discovering universal patterns

**Option C: Hybrid**
- Temporal edges within recordings (fixed)
- k-NN edges across recordings (learned)

**Recommendation:** Start with k-NN (simpler, adapts during training), then experiment with temporal edges if you want to enforce sequential structure.

---

## Post-Training Analysis

### Getting Hard Cluster Assignments

```python
# Soft assignments
C_soft = model.get_cluster_assignments(sequences)  # (N_seq, k)

# Hard assignments
cluster_ids = C_soft.argmax(dim=1)  # (N_seq,)

# Confidence scores
confidence = C_soft.max(dim=1).values  # (N_seq,)
```

### Analyzing Clusters

1. **Cluster sizes:** `torch.bincount(cluster_ids)`

2. **Cluster purity:** If you have any labels, compute how homogeneous each cluster is

3. **Cluster centroids:** Average sequence embedding per cluster
   ```python
   centroids = []
   for c in range(k):
       mask = (cluster_ids == c)
       centroids.append(S[mask].mean(dim=0))
   ```

4. **Characteristic sequences:** Find sequences closest to each centroid

5. **Temporal patterns:** For each recording, plot cluster assignments over time to see if certain clusters appear in bursts

6. **Transition matrix:** P(cluster_t+1 | cluster_t) to understand sequencing patterns

### Visualization

```python
# t-SNE/UMAP of sequence embeddings, colored by cluster
from sklearn.manifold import TSNE
S_2d = TSNE(n_components=2).fit_transform(S.detach().numpy())
plt.scatter(S_2d[:, 0], S_2d[:, 1], c=cluster_ids, cmap='tab20', s=1)
```

---

## Comparison with HDBSCAN Pipeline

| Aspect | GAT → HDBSCAN | GAT → DMoN (this architecture) |
|--------|---------------|-------------------------------|
| Clustering objective | Density-based | Modularity-based |
| Training | Two-stage | End-to-end |
| Embeddings optimized for | Reconstruction | Reconstruction + clustering |
| Number of clusters | Automatic | Fixed k (but soft assignments) |
| Handles noise | Yes (noise label) | No explicit noise class |
| Cluster shape | Arbitrary | Tends toward balanced |
| Scalability | O(N² log N) | O(N·k + edges) |

**When to prefer HDBSCAN:**
- You expect many noise/outlier sequences
- Clusters have very different sizes or densities
- You don't know how many clusters exist

**When to prefer DMoN:**
- You want end-to-end optimization
- You have a rough idea of cluster count
- You want soft/probabilistic assignments
- You care about graph structure (related sequences should cluster together)

---

## Summary

This architecture puts DMoN where it matters — at the sequence level:

1. **GAT** learns context-aware chirp embeddings using masked reconstruction
2. **Positional encodings + concatenation** preserve 4-gram structure
3. **Sequence MLP** compresses to a unified representation
4. **k-NN graph** captures sequence relationships
5. **DMoN** clusters sequences end-to-end

The entire pipeline is differentiable, so the GAT learns embeddings that ultimately produce good sequence clusters, not just good chirp reconstructions.
