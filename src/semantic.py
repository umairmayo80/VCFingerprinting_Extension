"""
semantic.py — Semantic distance evaluation for vcfp_reproduction.

Implements the two privacy metrics from the paper (Sec. II-C):
  1. Semantic Distance  D(Q, Q') = cosine_similarity(V_Q, V_Q')  [Eq. 3]
  2. Normalized Semantic Distance  NormDist(Q, Q', Q_set)        [Algorithm 1]

Two strategies for obtaining semantic vectors:
  - 'sentence_transformers': uses a pre-trained sentence embedding model (default)
  - 'doc2vec': trains a lightweight Gensim Doc2Vec model on a QA-like corpus

All reimplemented from scratch based solely on the paper's description.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ── Vector Store ───────────────────────────────────────────────────────────

VECTOR_CACHE_FILE = config.SEMANTIC_VECTORS_DIR / "command_vectors.pkl"


def save_vectors(vec_dict: Dict[str, np.ndarray]) -> None:
    """Persist the command→vector mapping to disk."""
    VECTOR_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(VECTOR_CACHE_FILE, "wb") as f:
        pickle.dump(vec_dict, f)
    print(f"[semantic] Saved {len(vec_dict)} vectors to {VECTOR_CACHE_FILE}")


def load_cached_vectors() -> Optional[Dict[str, np.ndarray]]:
    """Load previously cached command vectors, or return None."""
    if VECTOR_CACHE_FILE.exists():
        with open(VECTOR_CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return None


# ── Sentence-Transformers Strategy (default, no training needed) ────────────

def _humanize_command(cmd_str: str) -> str:
    """
    Convert a filename-style command string to a natural language sentence.
    Example: 'how_many_days_until_christmas' -> 'how many days until christmas'
    """
    return cmd_str.replace("_", " ")


def build_vectors_sentence_transformers(
    commands: List[str],
    model_name: str = config.SENTENCE_TRANSFORMER_MODEL,
) -> Dict[str, np.ndarray]:
    """
    Encode all voice commands using a pre-trained sentence transformer.
    Produces high-quality semantic vectors without any training.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. "
            "Run: pip install sentence-transformers"
        )

    print(f"[semantic] Loading sentence transformer: {model_name}")
    model = SentenceTransformer(model_name)

    sentences = [_humanize_command(c) for c in commands]
    print(f"[semantic] Encoding {len(sentences)} voice commands...")
    embeddings = model.encode(sentences, show_progress_bar=True, normalize_embeddings=True)

    vec_dict = {cmd: emb for cmd, emb in zip(commands, embeddings)}
    return vec_dict


# ── Doc2Vec Strategy (lightweight, trains on corpus) ──────────────────────

def _prepare_tagged_commands(commands: List[str]) -> list:
    """Create TaggedDocuments for the command strings themselves."""
    from gensim.models.doc2vec import TaggedDocument
    return [
        TaggedDocument(words=_humanize_command(cmd).lower().split(), tags=[cmd])
        for cmd in commands
    ]


def build_vectors_doc2vec(
    commands: List[str],
    corpus_sentences: Optional[List[str]] = None,
    vector_size: int = config.DOC2VEC_VECTOR_SIZE,
    epochs: int = config.DOC2VEC_EPOCHS,
    window: int = config.DOC2VEC_WINDOW,
    min_count: int = config.DOC2VEC_MIN_COUNT,
    seed: int = config.RANDOM_SEED,
) -> Dict[str, np.ndarray]:
    """
    Train a lightweight Doc2Vec model (DBOW) on:
      - The 100 voice command strings themselves
      - Optional additional corpus sentences (e.g., from a QA dataset)
    Then infer vectors for all 100 commands.

    If no corpus_sentences are given, the model is trained only on the
    100 command strings (gives reasonable but not ideal vectors).
    """
    try:
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    except ImportError:
        raise ImportError("gensim not installed. Run: pip install gensim")

    tagged_commands = _prepare_tagged_commands(commands)

    if corpus_sentences:
        # Also add corpus lines as additional training data
        extra_docs = [
            TaggedDocument(words=sent.lower().split(), tags=[f"__corpus_{i}"])
            for i, sent in enumerate(corpus_sentences)
        ]
        training_corpus = tagged_commands + extra_docs
    else:
        training_corpus = tagged_commands

    print(f"[semantic] Training Doc2Vec on {len(training_corpus)} documents "
          f"(vector_size={vector_size}, epochs={epochs})")

    model = Doc2Vec(
        documents=training_corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=1,
        dm=0,           # DBOW mode (same as original paper)
        dbow_words=1,
        negative=5,
        hs=0,
        epochs=epochs,
        seed=seed,
    )

    # Infer vectors for all 100 command strings
    vec_dict = {}
    for cmd in commands:
        words = _humanize_command(cmd).lower().split()
        vec = model.infer_vector(words, epochs=50, alpha=0.025)
        # L2-normalize
        norm = np.linalg.norm(vec)
        vec_dict[cmd] = vec / norm if norm > 0 else vec

    return vec_dict


# ── Main Entry Point ───────────────────────────────────────────────────────

def get_semantic_vectors(
    commands: List[str],
    strategy: str = config.SEMANTIC_STRATEGY,
    force_rebuild: bool = False,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    Return a dict mapping each command string to its semantic vector.

    1. Checks cache first (skip re-computation on subsequent runs).
    2. If not cached, builds vectors using the chosen strategy.
    3. Saves to cache before returning.
    """
    if not force_rebuild:
        cached = load_cached_vectors()
        if cached is not None:
            # Verify all requested commands are in cache
            missing = [c for c in commands if c not in cached]
            if not missing:
                print(f"[semantic] Loaded {len(cached)} cached vectors.")
                return cached
            print(f"[semantic] Cache missing {len(missing)} commands. Rebuilding.")

    if strategy == "sentence_transformers":
        vec_dict = build_vectors_sentence_transformers(commands, **kwargs)
    elif strategy == "doc2vec":
        vec_dict = build_vectors_doc2vec(commands, **kwargs)
    else:
        raise ValueError(f"Unknown semantic strategy: '{strategy}'. "
                         "Choose 'sentence_transformers' or 'doc2vec'.")

    save_vectors(vec_dict)
    return vec_dict


# ── Distance Metrics ───────────────────────────────────────────────────────

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    D(Q, Q') = (V_Q · V_Q') / (||V_Q|| · ||V_Q'||)   [Eq. 3 in paper]
    Returns value in [-1, 1].
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def normalized_semantic_distance(
    true_command: str,
    predicted_command: str,
    vec_dict: Dict[str, np.ndarray],
) -> int:
    """
    NormDistance(Q, Q', Q_set) — Algorithm 1 from the paper.

    Steps:
      1. Compute D(Q', Q_i) for all Q_i in the prior set Q_set.
      2. Sort these distances in DESCENDING order.
      3. Find the rank j (0-indexed) where D(Q, Q') appears.

    Return: rank j  (lower = better attack)
    Special case: if Q == Q' (correct prediction), rank = 0.

    Note: random guess expected value = 49.5 for M=100 classes.
    """
    if true_command == predicted_command:
        return 0  # perfect prediction, rank 0

    if predicted_command not in vec_dict or true_command not in vec_dict:
        return len(vec_dict) - 1  # worst case if vectors missing

    vec_pred = vec_dict[predicted_command]
    vec_true = vec_dict[true_command]

    # D(Q', Q_i) for all Q_i in the prior set except Q' itself
    similarities = []
    for cmd, vec in vec_dict.items():
        if cmd == predicted_command:
            continue
        sim = cosine_similarity(vec_pred, vec)
        similarities.append((cmd, sim))

    # Sort DESCENDING (highest similarity first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Find rank of the true command
    d_true = cosine_similarity(vec_pred, vec_true)
    for rank, (cmd, sim) in enumerate(similarities):
        if cmd == true_command:
            return rank  # 0-indexed rank

    # Fallback (should not happen)
    return len(similarities)


def evaluate_semantic_metrics(
    true_commands: List[str],
    predicted_commands: List[str],
    vec_dict: Dict[str, np.ndarray],
) -> Tuple[float, float]:
    """
    Compute aggregate semantic metrics over a list of predictions.

    Returns:
        mean_semantic_distance   : mean D(Q, Q') over all samples
        mean_normalized_distance : mean NormDist over all samples
    """
    assert len(true_commands) == len(predicted_commands), \
        "true_commands and predicted_commands must have the same length"

    semantic_dists = []
    norm_dists = []

    for true_cmd, pred_cmd in zip(true_commands, predicted_commands):
        # Cosine similarity
        if true_cmd in vec_dict and pred_cmd in vec_dict:
            sd = cosine_similarity(vec_dict[true_cmd], vec_dict[pred_cmd])
        else:
            sd = 0.0
        semantic_dists.append(sd)

        # Normalized semantic distance (Algorithm 1)
        nd = normalized_semantic_distance(true_cmd, pred_cmd, vec_dict)
        norm_dists.append(nd)

    return float(np.mean(semantic_dists)), float(np.mean(norm_dists))


# ── Sanity Check ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick test with 5 dummy commands
    test_cmds = [
        "what_is_the_weather",
        "what_is_the_weather_tomorrow",
        "how_many_days_until_christmas",
        "do_dogs_dream",
        "flip_a_coin",
    ]

    vec_dict = get_semantic_vectors(
        test_cmds,
        strategy="sentence_transformers",
        force_rebuild=True,
    )

    print("\nPairwise similarities:")
    for i, c1 in enumerate(test_cmds[:2]):
        for c2 in test_cmds:
            sim = cosine_similarity(vec_dict[c1], vec_dict[c2])
            print(f"  D('{c1}', '{c2}') = {sim:.4f}")

    nd = normalized_semantic_distance(
        "what_is_the_weather", "what_is_the_weather_tomorrow", vec_dict
    )
    print(f"\nNormDist('what_is_the_weather', 'what_is_the_weather_tomorrow') = {nd}")
