"""
GENREG Tree-of-Models Text Prediction Skeleton
================================================

Architecture:
  - Byte-level tokenizer (vocab=256, zero overhead)
  - Context encoder: evolved embeddings → context vector
  - Hierarchical routing tree: each node narrows the candidate token set
  - Leaves: small specialists that predict within a narrow token subset

Vocab 256, branching factor 4, depth 4 → 256 leaves (one per byte)
Vocab 256, branching factor 16, depth 2 → 256 leaves (flatter, wider cuts)

Each node is a small weight matrix — a genome that GENREG evolves.
No gradients anywhere. Every parameter is evolved.

HOOKS FOR GENREG:
  - TreeNode.weights → genome to evolve
  - TreeNode.evolve() → placeholder for GENREG evolution loop
  - ContextEncoder.embeddings → evolved embedding table
  - ContextEncoder.mix_weights → evolved context mixing
  - Full genome packing/unpacking for freeze-and-stack

Usage:
  1. Build tree with build_tree()
  2. Train bottom-up: evolve leaves first, freeze, evolve routers
  3. Inference: encode context → route through tree → predict token
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import struct


# ══════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════
@dataclass
class TreeConfig:
    vocab_size: int = 256            # byte-level
    branching_factor: int = 4        # how many children per node
    context_window: int = 32         # tokens of context
    embed_dim: int = 64              # embedding dimension
    context_dim: int = 128           # context vector size after mixing
    min_leaf_tokens: int = 1         # stop splitting when this few tokens remain
    max_depth: int = 8               # safety cap

    # Evolution params (defaults — GENREG overrides these)
    pop_size: int = 100
    generations: int = 200
    mutation_rate: float = 0.15
    elite_frac: float = 0.2

    @property
    def tree_depth(self):
        """Theoretical depth for full coverage."""
        import math
        return int(math.ceil(math.log(self.vocab_size) / math.log(self.branching_factor)))


# ══════════════════════════════════════════════════════════
#  CONTEXT ENCODER — Evolved embeddings + context mixing
# ══════════════════════════════════════════════════════════
class ContextEncoder:
    """
    Converts a window of token IDs into a fixed-size context vector.
    All parameters are evolved, not trained.

    Genome:
      - embeddings: (vocab_size × embed_dim) — evolved lookup table
      - mix_weights: (context_window × embed_dim, context_dim) — compresses context
      - mix_bias: (context_dim,)
    """
    def __init__(self, config: TreeConfig):
        self.config = config
        self.embeddings = np.random.randn(config.vocab_size, config.embed_dim) * 0.1
        self.mix_weights = np.random.randn(
            config.context_window * config.embed_dim, config.context_dim
        ) * 0.01
        self.mix_bias = np.zeros(config.context_dim)

        # Positional signal — fixed, not evolved (or evolve this too)
        self.positions = np.zeros((config.context_window, config.embed_dim))
        for i in range(config.context_window):
            for j in range(config.embed_dim):
                if j % 2 == 0:
                    self.positions[i, j] = np.sin(i / (10000 ** (j / config.embed_dim)))
                else:
                    self.positions[i, j] = np.cos(i / (10000 ** ((j-1) / config.embed_dim)))

    def encode(self, token_ids: np.ndarray) -> np.ndarray:
        """
        token_ids: (batch, context_window) — integer token IDs
        Returns: (batch, context_dim) — context vectors
        """
        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]

        batch_size = token_ids.shape[0]
        ctx_len = min(token_ids.shape[1], self.config.context_window)

        # Pad if shorter than context window
        if ctx_len < self.config.context_window:
            padded = np.zeros((batch_size, self.config.context_window), dtype=int)
            padded[:, -ctx_len:] = token_ids[:, -ctx_len:]
            token_ids = padded

        # Lookup embeddings + add position
        embeds = self.embeddings[token_ids]  # (batch, ctx_win, embed_dim)
        embeds = embeds + self.positions[np.newaxis, :, :]

        # Flatten and project
        flat = embeds.reshape(batch_size, -1)  # (batch, ctx_win * embed_dim)
        context = flat @ self.mix_weights + self.mix_bias  # (batch, context_dim)

        # Simple nonlinearity — tanh (no gradients needed, just shapes the space)
        context = np.tanh(context)
        return context

    @property
    def genome_size(self):
        return (self.config.vocab_size * self.config.embed_dim +
                self.config.context_window * self.config.embed_dim * self.config.context_dim +
                self.config.context_dim)

    def pack_genome(self) -> np.ndarray:
        """Flatten all parameters into a 1D genome vector."""
        return np.concatenate([
            self.embeddings.ravel(),
            self.mix_weights.ravel(),
            self.mix_bias.ravel()
        ])

    def unpack_genome(self, genome: np.ndarray):
        """Load parameters from a 1D genome vector."""
        cfg = self.config
        idx = 0

        n = cfg.vocab_size * cfg.embed_dim
        self.embeddings = genome[idx:idx+n].reshape(cfg.vocab_size, cfg.embed_dim)
        idx += n

        n = cfg.context_window * cfg.embed_dim * cfg.context_dim
        self.mix_weights = genome[idx:idx+n].reshape(
            cfg.context_window * cfg.embed_dim, cfg.context_dim)
        idx += n

        self.mix_bias = genome[idx:idx+cfg.context_dim]

    def freeze(self):
        """Mark as frozen — downstream can check this flag."""
        self._frozen = True

    @property
    def is_frozen(self):
        return getattr(self, '_frozen', False)


# ══════════════════════════════════════════════════════════
#  TREE NODE — Each node is a small evolved router or leaf
# ══════════════════════════════════════════════════════════
class TreeNode:
    """
    A single node in the routing tree.

    Router node:
      - Takes a context vector (context_dim,)
      - Outputs routing scores for each child (branching_factor,)
      - Routes to the child with highest score
      - Genome: weights (context_dim × branching_factor) + bias (branching_factor,)

    Leaf node:
      - Takes a context vector
      - Outputs prediction scores over its token subset
      - Genome: weights (context_dim × num_tokens) + bias (num_tokens,)
    """
    def __init__(self, node_id: str, token_set: List[int], config: TreeConfig, depth: int = 0):
        self.node_id = node_id
        self.token_set = token_set      # tokens this subtree is responsible for
        self.config = config
        self.depth = depth
        self.children: List['TreeNode'] = []
        self.is_leaf = False
        self._frozen = False

        # Will be set during build
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None

        # Stats
        self.routing_accuracy = 0.0
        self.prediction_accuracy = 0.0

    def init_as_router(self, num_children: int):
        """Initialize weights for routing decision."""
        self.weights = np.random.randn(self.config.context_dim, num_children) * 0.1
        self.bias = np.zeros(num_children)
        self.is_leaf = False

    def init_as_leaf(self, num_tokens: int):
        """Initialize weights for token prediction within subset."""
        self.weights = np.random.randn(self.config.context_dim, num_tokens) * 0.1
        self.bias = np.zeros(num_tokens)
        self.is_leaf = True

    def route(self, context: np.ndarray) -> np.ndarray:
        """
        context: (batch, context_dim) or (context_dim,)
        Returns: child indices (batch,)
        """
        if context.ndim == 1:
            context = context[np.newaxis, :]
        scores = context @ self.weights + self.bias  # (batch, num_children)
        return np.argmax(scores, axis=1)

    def predict(self, context: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Leaf prediction.
        context: (batch, context_dim)
        Returns: (predicted_token_indices_in_subset, raw_scores)
        """
        if context.ndim == 1:
            context = context[np.newaxis, :]
        scores = context @ self.weights + self.bias  # (batch, num_tokens_in_subset)
        local_idx = np.argmax(scores, axis=1)
        return local_idx, scores

    @property
    def genome_size(self):
        if self.weights is None:
            return 0
        return self.weights.size + self.bias.size

    def pack_genome(self) -> np.ndarray:
        return np.concatenate([self.weights.ravel(), self.bias.ravel()])

    def unpack_genome(self, genome: np.ndarray):
        w_size = self.weights.size
        self.weights = genome[:w_size].reshape(self.weights.shape)
        self.bias = genome[w_size:]

    def freeze(self):
        self._frozen = True

    @property
    def is_frozen(self):
        return self._frozen


# ══════════════════════════════════════════════════════════
#  TREE BUILDER
# ══════════════════════════════════════════════════════════
def build_tree(config: TreeConfig) -> TreeNode:
    """
    Build a balanced routing tree over the vocabulary.

    Splits token set by branching_factor at each level.
    Stops when a node covers <= min_leaf_tokens tokens or max_depth reached.
    """
    all_tokens = list(range(config.vocab_size))
    root = _build_recursive(all_tokens, config, depth=0, path="R")
    return root


def _build_recursive(tokens: List[int], config: TreeConfig, depth: int, path: str) -> TreeNode:
    node = TreeNode(node_id=path, token_set=tokens, config=config, depth=depth)

    # Leaf condition
    if len(tokens) <= config.min_leaf_tokens or depth >= config.max_depth:
        node.init_as_leaf(max(len(tokens), 1))
        return node

    # Split tokens into branching_factor groups
    bf = min(config.branching_factor, len(tokens))
    chunk_size = len(tokens) // bf
    remainder = len(tokens) % bf

    chunks = []
    idx = 0
    for i in range(bf):
        size = chunk_size + (1 if i < remainder else 0)
        chunks.append(tokens[idx:idx+size])
        idx += size

    # Remove empty chunks
    chunks = [c for c in chunks if len(c) > 0]
    actual_bf = len(chunks)

    if actual_bf <= 1:
        # Can't split further
        node.init_as_leaf(len(tokens))
        return node

    node.init_as_router(actual_bf)

    for i, chunk in enumerate(chunks):
        child = _build_recursive(chunk, config, depth + 1, f"{path}.{i}")
        node.children.append(child)

    return node


# ══════════════════════════════════════════════════════════
#  TREE INFERENCE
# ══════════════════════════════════════════════════════════
def tree_predict_token(root: TreeNode, encoder: ContextEncoder,
                       token_ids: np.ndarray) -> np.ndarray:
    """
    Full inference: context → route through tree → predict next token.

    token_ids: (batch, seq_len) or (seq_len,)
    Returns: (batch,) predicted next token IDs
    """
    context = encoder.encode(token_ids)  # (batch, context_dim)
    batch_size = context.shape[0]
    predictions = np.zeros(batch_size, dtype=int)

    # Route each sample through the tree
    # (Batched routing where possible, sample-level at branch points)
    _route_batch(root, context, np.arange(batch_size), predictions)

    return predictions


def _route_batch(node: TreeNode, context: np.ndarray,
                 indices: np.ndarray, predictions: np.ndarray):
    """Recursively route a batch of samples through the tree."""
    if len(indices) == 0:
        return

    ctx_subset = context[indices]

    if node.is_leaf:
        # Predict within this node's token subset
        local_idx, _ = node.predict(ctx_subset)
        # Map local index back to global token ID
        for i, sample_idx in enumerate(indices):
            token_subset_idx = min(local_idx[i], len(node.token_set) - 1)
            predictions[sample_idx] = node.token_set[token_subset_idx]
        return

    # Route to children
    child_choices = node.route(ctx_subset)  # (len(indices),)

    # Group samples by chosen child
    for c_idx, child in enumerate(node.children):
        mask = child_choices == c_idx
        if np.any(mask):
            _route_batch(child, context, indices[mask], predictions)


# ══════════════════════════════════════════════════════════
#  TREE TRAINING — Bottom-up freeze-and-stack
# ══════════════════════════════════════════════════════════
def collect_training_data(text: str, config: TreeConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert raw text to (context_windows, next_token) pairs.
    Byte-level tokenization — no tokenizer overhead.
    """
    tokens = np.array(list(text.encode('utf-8')), dtype=int)

    if len(tokens) <= config.context_window:
        return np.array([]).reshape(0, config.context_window), np.array([])

    X = []
    y = []
    for i in range(config.context_window, len(tokens)):
        X.append(tokens[i - config.context_window:i])
        y.append(tokens[i])

    return np.array(X), np.array(y)


def train_tree_bottom_up(root: TreeNode, encoder: ContextEncoder,
                         X: np.ndarray, y: np.ndarray, config: TreeConfig,
                         evolve_fn=None):
    """
    Bottom-up training:
    1. Evolve encoder (or use pre-evolved one)
    2. Encode all training contexts
    3. For each leaf: evolve specialist on its token subset's data
    4. Freeze leaves
    5. For each router (bottom to top): evolve on routing labels
    6. Freeze each router after training

    evolve_fn: GENREG evolution hook
      signature: evolve_fn(genome_size, fitness_fn, config) → best_genome
      If None, uses built-in simple evolution.
    """
    if evolve_fn is None:
        evolve_fn = _default_evolve

    # Step 1: Encode all contexts
    # (Encoder should be pre-evolved or evolved here)
    contexts = encoder.encode(X)  # (N, context_dim)
    N = len(y)

    print(f"Training tree on {N} samples")
    print(f"Tree config: vocab={config.vocab_size}, branch={config.branching_factor}, "
          f"context_dim={config.context_dim}")

    # Step 2: Collect per-node training sets, then evolve bottom-up
    _train_node_recursive(root, contexts, y, config, evolve_fn, total_samples=N)

    print("\nTraining complete. All nodes frozen.")


def _train_node_recursive(node: TreeNode, contexts: np.ndarray, targets: np.ndarray,
                          config: TreeConfig, evolve_fn, total_samples: int):
    """Recursively train nodes bottom-up."""

    # Filter to samples whose target is in this node's token set
    token_set = set(node.token_set)
    mask = np.array([t in token_set for t in targets])
    ctx_local = contexts[mask]
    tgt_local = targets[mask]

    if len(ctx_local) == 0:
        print(f"  {'  '*node.depth}Node {node.node_id}: 0 samples, skipping")
        return

    coverage = len(ctx_local) / total_samples * 100

    if node.is_leaf:
        # ── EVOLVE LEAF SPECIALIST ──
        # Map global token IDs to local indices
        token_to_local = {t: i for i, t in enumerate(node.token_set)}
        local_targets = np.array([token_to_local.get(t, 0) for t in tgt_local])
        num_classes = len(node.token_set)

        def leaf_fitness(genome):
            node.unpack_genome(genome)
            local_pred, _ = node.predict(ctx_local)
            return np.mean(local_pred == local_targets)

        best = evolve_fn(node.genome_size, leaf_fitness, config)
        node.unpack_genome(best)
        acc = np.mean(node.predict(ctx_local)[0] == local_targets)
        node.prediction_accuracy = acc
        node.freeze()

        print(f"  {'  '*node.depth}Leaf {node.node_id}: "
              f"{len(node.token_set)} tokens, {len(ctx_local)} samples ({coverage:.1f}%), "
              f"acc={acc:.3f} ✓ frozen")
        return

    # ── RECURSE INTO CHILDREN FIRST (bottom-up) ──
    for child in node.children:
        _train_node_recursive(child, contexts, targets, config, evolve_fn, total_samples)

    # ── EVOLVE ROUTER ──
    # Labels: which child should each sample go to?
    child_sets = [set(c.token_set) for c in node.children]
    routing_labels = np.zeros(len(tgt_local), dtype=int)
    for i, t in enumerate(tgt_local):
        for c_idx, cs in enumerate(child_sets):
            if t in cs:
                routing_labels[i] = c_idx
                break

    def router_fitness(genome):
        node.unpack_genome(genome)
        chosen = node.route(ctx_local)
        return np.mean(chosen == routing_labels)

    best = evolve_fn(node.genome_size, router_fitness, config)
    node.unpack_genome(best)
    acc = np.mean(node.route(ctx_local) == routing_labels)
    node.routing_accuracy = acc
    node.freeze()

    print(f"  {'  '*node.depth}Router {node.node_id}: "
          f"{len(node.children)} children, {len(ctx_local)} samples ({coverage:.1f}%), "
          f"routing_acc={acc:.3f} ✓ frozen")


# ══════════════════════════════════════════════════════════
#  DEFAULT EVOLUTION — Replace with GENREG
# ══════════════════════════════════════════════════════════
def _default_evolve(genome_size: int, fitness_fn, config: TreeConfig) -> np.ndarray:
    """
    Simple GA placeholder. GENREG replaces this entirely.

    GENREG HOOK: Replace this function with your evolution engine.
    Interface contract:
      Input: genome_size (int), fitness_fn(genome) → float, config
      Output: best genome as np.ndarray of shape (genome_size,)
    """
    pop = np.random.randn(config.pop_size, genome_size) * 0.1
    best_score = -1
    best_ind = pop[0]
    top_k = max(2, int(config.pop_size * config.elite_frac))

    for gen in range(config.generations):
        scores = np.array([fitness_fn(ind) for ind in pop])
        idx = np.argmax(scores)
        if scores[idx] > best_score:
            best_score = scores[idx]
            best_ind = pop[idx].copy()

        elites = pop[np.argsort(scores)[-top_k:]]
        new_pop = [best_ind.copy()]
        for _ in range(config.pop_size - 1):
            p1, p2 = elites[np.random.choice(len(elites), 2, replace=False)]
            mask = np.random.rand(genome_size) > 0.5
            child = np.where(mask, p1, p2)
            if np.random.rand() < config.mutation_rate:
                # Relative mutation — GENREG style
                scale = np.abs(child) * 0.1 + 0.01
                child += np.random.randn(genome_size) * scale
            new_pop.append(child)
        pop = np.array(new_pop)

    return best_ind


# ══════════════════════════════════════════════════════════
#  SERIALIZATION — Save/Load frozen trees
# ══════════════════════════════════════════════════════════
def save_tree(root: TreeNode, encoder: ContextEncoder, path: str):
    """Serialize the full model (encoder + tree) to a .npz file."""
    data = {"encoder_genome": encoder.pack_genome()}

    def _save_node(node, prefix):
        data[f"{prefix}_tokens"] = np.array(node.token_set)
        data[f"{prefix}_is_leaf"] = np.array([node.is_leaf])
        data[f"{prefix}_weights"] = node.weights
        data[f"{prefix}_bias"] = node.bias
        data[f"{prefix}_depth"] = np.array([node.depth])
        if not node.is_leaf:
            data[f"{prefix}_num_children"] = np.array([len(node.children)])
            for i, child in enumerate(node.children):
                _save_node(child, f"{prefix}_c{i}")

    _save_node(root, "root")
    np.savez_compressed(path, **data)


def load_tree(path: str, config: TreeConfig) -> Tuple[TreeNode, ContextEncoder]:
    """Deserialize a frozen model."""
    data = np.load(path, allow_pickle=True)

    encoder = ContextEncoder(config)
    encoder.unpack_genome(data["encoder_genome"])

    def _load_node(prefix):
        tokens = data[f"{prefix}_tokens"].tolist()
        is_leaf = bool(data[f"{prefix}_is_leaf"][0])
        depth = int(data[f"{prefix}_depth"][0])
        node = TreeNode(prefix, tokens, config, depth)
        node.weights = data[f"{prefix}_weights"]
        node.bias = data[f"{prefix}_bias"]
        node.is_leaf = is_leaf
        node._frozen = True
        if not is_leaf:
            nc = int(data[f"{prefix}_num_children"][0])
            for i in range(nc):
                child = _load_node(f"{prefix}_c{i}")
                node.children.append(child)
        return node

    root = _load_node("root")
    return root, encoder


# ══════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════
def evaluate(root: TreeNode, encoder: ContextEncoder,
             X: np.ndarray, y: np.ndarray) -> Dict:
    """
    Full evaluation of the tree model.
    Returns accuracy, per-depth routing stats, throughput.
    """
    import time

    t0 = time.time()
    preds = tree_predict_token(root, encoder, X)
    elapsed = time.time() - t0

    accuracy = np.mean(preds == y)
    tokens_per_sec = len(y) / max(elapsed, 1e-6)

    # Per-depth stats
    depth_stats = {}
    _collect_depth_stats(root, depth_stats)

    return {
        "accuracy": accuracy,
        "tokens_per_sec": tokens_per_sec,
        "num_samples": len(y),
        "depth_stats": depth_stats,
        "unique_predictions": len(np.unique(preds)),
        "target_entropy": _entropy(y),
    }


def _collect_depth_stats(node, stats):
    d = node.depth
    if d not in stats:
        stats[d] = {"nodes": 0, "routers": 0, "leaves": 0, "avg_acc": []}
    stats[d]["nodes"] += 1
    if node.is_leaf:
        stats[d]["leaves"] += 1
        stats[d]["avg_acc"].append(node.prediction_accuracy)
    else:
        stats[d]["routers"] += 1
        stats[d]["avg_acc"].append(node.routing_accuracy)
        for child in node.children:
            _collect_depth_stats(child, stats)


def _entropy(arr):
    _, counts = np.unique(arr, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-12))


# ══════════════════════════════════════════════════════════
#  TREE DIAGNOSTICS
# ══════════════════════════════════════════════════════════
def print_tree_summary(root: TreeNode, config: TreeConfig):
    """Print a human-readable summary of the tree structure."""
    total_params = 0
    total_nodes = 0
    depth_count = {}

    def _walk(node):
        nonlocal total_params, total_nodes
        total_nodes += 1
        total_params += node.genome_size
        d = node.depth
        depth_count[d] = depth_count.get(d, 0) + 1
        for child in node.children:
            _walk(child)

    _walk(root)

    print(f"\n{'═'*50}")
    print(f"  TREE SUMMARY")
    print(f"{'═'*50}")
    print(f"  Total nodes:      {total_nodes}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Vocab size:       {config.vocab_size}")
    print(f"  Branching factor: {config.branching_factor}")
    print(f"  Context window:   {config.context_window}")
    print(f"  Context dim:      {config.context_dim}")
    print(f"  Embed dim:        {config.embed_dim}")
    for d in sorted(depth_count.keys()):
        print(f"  Depth {d}: {depth_count[d]} nodes")
    print(f"{'═'*50}")


# ══════════════════════════════════════════════════════════
#  DEMO — Quick sanity check
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  GENREG Tree-of-Models — Skeleton Demo")
    print("=" * 60)

    # Small config for demo
    cfg = TreeConfig(
        vocab_size=256,
        branching_factor=4,
        context_window=16,
        embed_dim=32,
        context_dim=64,
        pop_size=50,
        generations=50,  # low for demo speed
    )

    # Build tree
    print("\nBuilding tree...")
    root = build_tree(cfg)
    print_tree_summary(root, cfg)

    # Training data — just repeat a simple pattern for demo
    text = "the cat sat on the mat. the dog sat on the log. " * 100
    X, y = collect_training_data(text, cfg)
    print(f"\nTraining data: {len(X)} samples from {len(text)} bytes")

    # Encoder
    encoder = ContextEncoder(cfg)

    # Train
    print("\nTraining (bottom-up freeze-and-stack)...")
    train_tree_bottom_up(root, encoder, X, y, cfg)

    # Evaluate
    results = evaluate(root, encoder, X, y)
    print(f"\n{'═'*50}")
    print(f"  RESULTS")
    print(f"{'═'*50}")
    print(f"  Accuracy:        {results['accuracy']:.4f}")
    print(f"  Tokens/sec:      {results['tokens_per_sec']:.0f}")
    print(f"  Unique preds:    {results['unique_predictions']}")
    print(f"  Target entropy:  {results['target_entropy']:.2f} bits")

    # Bigram baseline
    from collections import Counter
    bigram_correct = 0
    bigram_counts = Counter()
    for i in range(len(y)):
        prev = X[i, -1]
        bigram_counts[prev] = bigram_counts.get(prev, Counter())
    # Simplified: most common next token per last token
    bigram_table = {}
    for i in range(len(y)):
        prev = int(X[i, -1])
        if prev not in bigram_table:
            bigram_table[prev] = Counter()
        bigram_table[prev][int(y[i])] += 1
    bigram_preds = np.array([bigram_table.get(int(X[i, -1]), Counter()).most_common(1)[0][0]
                             if bigram_table.get(int(X[i, -1])) else 0
                             for i in range(len(y))])
    bigram_acc = np.mean(bigram_preds == y)
    print(f"  Bigram baseline: {bigram_acc:.4f}")
    print(f"  vs Tree:         {'+' if results['accuracy'] > bigram_acc else ''}"
          f"{results['accuracy'] - bigram_acc:+.4f}")

    # Save
    save_path = "/tmp/tree_model_demo.npz"
    save_tree(root, encoder, save_path)
    print(f"\n  Model saved to {save_path}")
    print(f"  Done.")
