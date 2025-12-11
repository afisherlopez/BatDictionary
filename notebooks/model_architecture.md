# Model Architecture

I'm using a Graph Attention Transformer to learn embeddings for k-grams of chirps. Right now, I'm setting k=4 for preliminary testing. I have 41,774 chirps across 2,929 recordings.

---

## 1. Preprocess & Create Graphs

- Reduce to PCA components
- Normalize features
- For each recording:
  - Create a node feature matrix X_r ∈ ℝ^(n_r × 30)
  - Create edge indices and edge feature arrays
    - Suggestion: e_{i,i+1} = f(Δt) = log(1 + Δt), then normalize over the whole dataset
    - **Open question:** Why not just normalize the time intervals (already a feature) and just have those raw time intervals be the edge features?
    - **Open question:** I feel like the graph should be directed because time moves forward, so a chirp cannot be caused by the one following it. But Chat says undirected/bidirectional.
    - If undirected: (n_r - 1) edges
    - If bidirectional: 2(n_r - 1) edges

---

## 2. GAT Architecture

To create contextual embeddings for each node (embeddings which take into account the surrounding nodes), I'm creating a one-dimensional graph for each recording in which each temporally adjacent chirp is linked with an edge which represents the time interval between the chirps, and each chirp is represented as a node with a feature vector x ∈ ℝ^30. I then train a two-layer Graph Attention Transformer on the entire file to produce context-rich embeddings h_c ∈ ℝ^32 for each chirp.

**Open questions:**
- How can we ensure that the embeddings are the same structure across recordings so that all sequences can be meaningfully clustered later?
- Should we share W_Q, W_K, W_V, etc. between all recordings?
- How can we train those matrices to achieve best performance when creating context-rich embeddings across all recordings?

### First Layer

**Input:** x_i ∈ ℝ^30 (PCA component for an individual chirp)

**For each of the 4 heads** (open question: how do the heads correspond to adjacent chirps?):

1. **Linear projections for chirps:**
   - q_i^(h) = W_Q^(h) x_i
   - k_i^(h) = W_K^(h) x_i
   - v_i^(h) = W_V^(h) x_i
   - where W_Q^(h), W_K^(h), W_V^(h) ∈ ℝ^(8×30)

2. **MLP for edge feature:**
   - φ_e : ℝ → ℝ^4
   - Creates edge features z_{ij} = φ_e(e_{ij}) ∈ ℝ^4

3. **Attention logits:**
   - ℓ_{ij}^(h) = u^(h) · LeakyReLU([q_i^(h) || k_j^(h) || z_{ij}])
   - where u^(h) ∈ ℝ^(8+8+4) is a learnable vector

4. **Compute normalized attention:**
   - α_{ij}^(h) = exp(ℓ_{ij}^(h)) / Σ_{k∈N(i)} exp(ℓ_{ik}^(h))

5. **Message aggregation:**
   - h̃_i^(h) = Σ_{j∈N(i)} α_{ij}^(h) v_j^(h)

6. **Concatenate heads:**
   - h̃_i = [h̃_i^(1) || h̃_i^(2) || h̃_i^(3) || h̃_i^(4)] ∈ ℝ^32

7. **Apply nonlinearity and optional dropout to h̃_i**

### Second Layer

Same structure as first layer, but input dimension is 32, so:
- W_Q^(h), W_K^(h), W_V^(h) ∈ ℝ^(8×32)

**Final output:** h̃_{i,final} ∈ ℝ^32

### Global Suggestions

- Dropout on node features and attention coefficients (e.g., 0.1 - 0.2)
- L2 weight decay (e.g., 10^(-5))

---

## 3. Self-supervised Training

Since our embeddings are completely unsupervised and we have no labels, we want to induce some sort of self-supervision so that GAT does not just produce trivial embeddings. To do this, we will randomly choose 10% of chirps from each subgraph, and make their input features noisy by replacing the PCA components with Gaussian noise (or 0 if that doesn't work). Then, after GAT processes the graph, we'll use a Multilayer Perceptron (MLP) to reconstruct those chirps from their final embeddings, minimizing the loss function:

**L_MSE = (1/|M|) Σ_{i∈M} ||x̂_i - x_i||_2^2**

This will force the learned embeddings to contain enough information from their neighbors that the MLP should be able to reconstruct them. If it can't, that's an initial indicator that there isn't enough relevant contextual information to draw strong connections between chirps.

### Masking

- For each node i ∈ X, sample a Bernoulli mask with p = 0.1
- Let M be the set of masked nodes
- For each i ∈ M, replace x_i with x̃_i ~ N(0, σI_30) where σ is the global standard deviation of the PCA components
  - (If this leads to training instability, can just set x̃_i = 0)
- For unmasked nodes, x̃_i = x_i
- Run the GAT on X̃ to get embeddings h_i for all nodes

### Backprop & Train End-to-end

**Open question:** What does it mean when it says "train end-to-end"?

- Define a decoder MLP f : ℝ^32 → ℝ^30
  - Example architecture: ReLU hidden layer 32 → 64, linear output layer 64 → 30
- For every masked node i ∈ M, get predicted embedding x̂_i = f(h_i)
- Then, backpropagate through the MLP, GAT, and the edge decoder φ_e to optimize the training loss:

**L_MSE = (1/|M|) Σ_{i∈M} ||x̂_i - x_i||_2^2**

- Use Adam as an optimizer
- Learning rate: 10^(-3)
- Batch size: one recording per batch
- 50 epochs with early stopping based on validation reconstruction loss

---

## 4a. Add Positional Encodings

**Open question:** Should we do this now, to capture position within the entire recording, or later, to capture position inside the sequence?

---

## 4b. Creating Sequence Embeddings

Once we have the chirp embeddings, we use learned positional embeddings to ensure that temporal structure is being preserved. We concatenate these positional embeddings to the GAT embeddings, giving us:

**h'_c = h_c + p_c**

for each chirp, where p_c is the positional encoding for that chirp.

We then create embeddings for each sequence by concatenating the embeddings for the four chirps together:

**h_s = [h'_1 || h'_2 || h'_3 || h'_4]**

We then pass these embeddings through an MLP (or another GAT if we're being fancy but probably unnecessary) to create fused embeddings which are not agnostic to the positions of each chirp within the sequence. The MLP takes advantage of the positional encodings we added earlier to add more positional information within the sequence embeddings.

**Final embeddings for each sequence:**

**h'_s = MLP(h_s) ∈ ℝ^32**

---

## GAT w/ built-in Deep Modularity Networks (DMoN)

I'll create a wider graph of all of the 4-grams by concatenating the embeddings from the previous GAT to create one embedding for each sequence.

---

## Summary of Dimensions

| Stage | Tensor | Dimensions |
|-------|--------|------------|
| Input chirp features | x | ℝ^30 |
| GAT chirp embeddings | h_c | ℝ^32 |
| Chirp + positional | h'_c | ℝ^32 |
| Concatenated 4-gram | h_s | ℝ^128 (4 × 32) |
| Final sequence embedding | h'_s | ℝ^32 |

---

## Open Questions Summary

1. Why not just normalize time intervals and use raw values as edge features?
2. Should the graph be directed (causal) or undirected/bidirectional?
3. How do the 4 attention heads correspond to adjacent chirps?
4. How to ensure embedding consistency across recordings for meaningful clustering?
5. Should W_Q, W_K, W_V be shared across all recordings?
6. What does "train end-to-end" mean in this context?
7. When to add positional encodings: now (recording-level) or later (sequence-level)?
