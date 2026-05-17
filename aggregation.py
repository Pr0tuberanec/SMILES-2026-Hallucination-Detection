"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

from model import MAX_LENGTH, _DEFAULT_MODEL

HIDDEN_DIM = 896
USE_GEOMETRIC = True

SELECTED_HEADS = [
    (18, "mean"),
    (21, "mean"),
    (19, "mean"),
    (22, "mean"),
    (9, "mean"),
    (7, "mean"),
    (15, "max"),
    (12, "mean"),
]

_DATA_DIR = Path(__file__).resolve().parent / "data"
_TRAIN_CSV = _DATA_DIR / "dataset.csv"
_TEST_CSV = _DATA_DIR / "test.csv"
_N_TRAIN = 689

_tokenizer = None
_train_prompts = None
_test_prompts = None
_sample_counter = 0


def reset_sample_counter():
    global _sample_counter
    _sample_counter = 0


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    pool_mask=None,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the aggregation below.
    # ------------------------------------------------------------------

    if pool_mask is None:
        pool_mask = _pooling_mask(attention_mask, hidden_states)
    blocks = []
    for layer, pool_type in SELECTED_HEADS:
        blocks.append(_pool_one(hidden_states[layer], pool_mask, pool_type))
    return torch.cat(blocks, dim=0)
    # ------------------------------------------------------------------


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    pool_mask=None,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------

    if pool_mask is None:
        pool_mask = _pooling_mask(attention_mask, hidden_states)
    head_vecs = _head_vectors_for_selected(hidden_states, pool_mask)
    feats = []

    for layer, _pool_type in SELECTED_HEADS:
        tokens = hidden_states[layer][pool_mask.to(hidden_states.device)]
        if tokens.shape[0] <= 1:
            feats.append(0.0)
        else:
            feats.append(tokens.norm(dim=-1).std().item())

    indexed = sorted(zip(SELECTED_HEADS, head_vecs), key=lambda item: item[0][0])
    for i in range(len(indexed) - 1):
        v1 = indexed[i][1]
        v2 = indexed[i + 1][1]
        n1 = v1.norm().clamp(min=1e-8)
        n2 = v2.norm().clamp(min=1e-8)
        feats.append((n2 / n1).item())
        feats.append(_cosine_similarity(v1, v2).item())

    head_norms = torch.stack([v.norm() for v in head_vecs])
    feats.append(head_norms.std().item())
    feats.append(head_norms.mean().item())
    feats.append(float(pool_mask.sum().item()))
    if len(head_vecs) >= 2:
        cosines = [
            _cosine_similarity(head_vecs[i], head_vecs[j]).item()
            for i in range(len(head_vecs))
            for j in range(i + 1, len(head_vecs))
        ]
        feats.append(float(sum(cosines) / len(cosines)))
    else:
        feats.append(0.0)

    return torch.tensor(feats, dtype=torch.float32, device=hidden_states.device)
    # ------------------------------------------------------------------


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    pool_mask = _pooling_mask(attention_mask, hidden_states)
    agg_features = aggregate(hidden_states, attention_mask, pool_mask=pool_mask)
    if USE_GEOMETRIC:
        geo_features = extract_geometric_features(
            hidden_states, attention_mask, pool_mask=pool_mask
        )
        return torch.cat([agg_features, geo_features], dim=0)
    return agg_features


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(_DEFAULT_MODEL)
    return _tokenizer


def _load_prompts():
    global _train_prompts, _test_prompts
    if _train_prompts is None:
        _train_prompts = pd.read_csv(_TRAIN_CSV)["prompt"].tolist()
    if _test_prompts is None:
        _test_prompts = pd.read_csv(_TEST_CSV)["prompt"].tolist()
    return _train_prompts, _test_prompts


def _prompt_for_current_sample():
    global _sample_counter
    train_prompts, test_prompts = _load_prompts()
    if _sample_counter < _N_TRAIN:
        prompt = train_prompts[_sample_counter]
    else:
        prompt = test_prompts[_sample_counter - _N_TRAIN]
    _sample_counter += 1
    return prompt


def _prompt_token_length(prompt):
    return len(
        _get_tokenizer()(prompt, truncation=True, max_length=MAX_LENGTH)["input_ids"]
    )


def _pool_one(layer_states, pool_mask, pool_type):
    masked = layer_states[pool_mask.to(layer_states.device)]
    if masked.numel() == 0:
        return torch.zeros(HIDDEN_DIM, dtype=layer_states.dtype, device=layer_states.device)
    if pool_type == "mean":
        return masked.mean(dim=0)
    return masked.max(dim=0).values


def _cosine_similarity(a, b, eps=1e-8):
    return torch.dot(a, b) / (a.norm() * b.norm() + eps)


def _head_vectors_for_selected(hidden_states, pool_mask):
    return [_pool_one(hidden_states[layer], pool_mask, pool_type) for layer, pool_type in SELECTED_HEADS]


def _pooling_mask(attention_mask, hidden_states):
    real = attention_mask.bool()
    start = min(_prompt_token_length(_prompt_for_current_sample()), real.shape[0])
    mask = real.clone()
    mask[:start] = False
    if not mask.any():
        mask = real
    if mask.any():
        last = int(mask.nonzero(as_tuple=False)[-1].item())
        mask[last] = False
        if not mask.any():
            mask = real.clone()
            mask[:start] = False
    return mask.to(hidden_states.device)


reset_sample_counter()
