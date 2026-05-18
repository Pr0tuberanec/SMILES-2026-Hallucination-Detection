# SMILES-2026-Hallucination Detection

## 1. Reproducibility

### Environment

- **Python 3.10+**
- **GPU recommended:** feature extraction with Qwen2.5-0.5B on 689 train + 100 test samples (~3 min on Colab T4)
- **Dependencies:** `requirements.txt` (`torch`, `transformers`, `scikit-learn`, `xgboost`)

### Commands

From the repository root:

```bash
pip install -r requirements.txt
python solution.py
```

---

## 2. Final solution

| File | Changes |
|------|---------|
| `aggregation.py` | 8 layers for hidden-state pooling (details below); pooling on the answer only, without the last token (EOS); +26 geometric features |
| `probe.py` | XGBoost |
| `splitting.py` | Stratified 5-fold cross-validation |

**Final pipeline:** Qwen2.5-0.5B → 7194-dim features → XGBoost → **test AUROC 77.86%**

Below are the experiments behind each design choice

---

### 2.1. Layer and pooling selection

**Goal:** pick a subset of 48 candidates (24 layers × mean/max) that carries a hallucination signal, and reduce size from 43008 (48×896) to something workable

**My approach:** for each (layer, pooling) pair, I train a separate logistic regression on the 896-d pooled vector.
I then collect out-of-fold probabilities over 5 folds and train a meta logistic regression on top of them. I take the top 8 poolings by absolute meta weights (the meta model down-weights weak poolings by giving them small coefficients)

**Results (selected poolings):**

```python
SELECTED_HEADS = [
    (18, "mean"), (21, "mean"), (19, "mean"), (22, "mean"),
    (9, "mean"), (7, "mean"), (15, "max"), (12, "mean"),
]
```

| layer | pool | \|w_meta\| |
|------:|------|----------:|
| 18 | mean | 0.126 |
| 21 | mean | 0.116 |
| 19 | mean | 0.105 |
| 22 | mean | 0.086 |
| 9 | mean | 0.084 |
| 7 | mean | 0.081 |
| 15 | max | 0.081 |
| 12 | mean | 0.053 |

**Why this worked:** the meta model relies mostly on **middle and late layers** (7 to 22), almost all **mean pooling**, and one **max** (layer 15). Mean pooling captures the overall representation of the answer across the sequence. For hallucination that seems more stable than a summary built from local spikes. This is consistent with the idea that factuality-related information may be encoded closer to the top of the network. Eight heads (7168 dim) are a trade-off between information and scale

---

### 2.2. Token mask for pooling

**Goal:** see which tokens to pool over: the full sequence, only the assistant’s answer, or the answer without the last token (EOS). Fixed `SELECTED_HEADS`, default MLP probe, 7168 dim, 5-fold CV

**Results** (test split, mean over 5 folds):

| Mask | Test Acc | Test F1 | Test AUROC |
|------|----------|---------|------------|
| 1. Full sequence (prompt + answer + EOS) | 70.39% | 80.78% | 73.33% |
| 2. Answer tokens only | 71.12% | 81.49% | 73.34% |
| 3. Answer without last token (no EOS) | 70.97% | 81.45% | **73.57%** |

**Why this worked:** dropping the prompt (variant 2) slightly raised F1, since the hallucination signal seems stronger in the **generated text** than in the question. Dropping EOS (variant 3) gave the **best test AUROC** (+0.24 pp vs variant 1): the last token is often `<|endoftext|>`, and mean/max over it dilutes the meaningful answer tokens

---

### 2.3. Geometric features

**Goal:** add hand-crafted stats to the pooled vector (7168 dim): token norm std on selected layers, norm ratios and cosines between neighboring layers, global aggregates (std and mean of norms over all 8 pooled vectors). Total +26 scalars → 7194 dim. Same mask as in §2.2 (variant 3), MLP, 5-fold CV

**Results:**

| Checkpoint | Accuracy | F1 | AUROC |
|------------|----------|-----|-------|
| Majority baseline | 70.10% | 82.42% | n/a |
| Probe (train) | 88.86% | 93.38% | 100.00% |
| Probe (val) | 73.27% | 83.20% | 74.15% |
| **Probe (test)** | **71.26%** | **82.07%** | **73.59%** |

Without geometry (pooling only): test AUROC **73.57%**

**Why this didn't work:** I added 26 scalars to describe the **shape** of the representation: spread of activations within a layer (token norm std), “drift” between neighboring layers (norm ratios and cosines), overall scale of all 8 heads (mean/std of norms, mean pairwise cosine, answer length in tokens). Test AUROC went up only **+0.02 pp** (73.57% → 73.59%), **no improvement**. The added geometric statistics may contain some signal, but it is likely too small relative to the high-dimensional pooled representation. The network **does not really know how to use** these numbers in the decision. I kept the block in the final run: it does not hurt the metric and does not require re-running the LLM, and it might help in other setups

---

### 2.4. Classifier (probe)

**Goal:** on one feature matrix (mask from §2.2 variant 3 + geometry, 7194 dim) compare classifiers and pick the best by test AUROC

**Results** (5-fold, test, mean):

| Model | Test Acc | Test F1 | Test AUROC |
|-------|----------|---------|------------|
| Logistic regression | 71.26% | 81.50% | 72.38% |
| MLP (256) | 70.24% | 81.29% | 73.71% |
| MLP + regularization | 70.97% | 82.13% | 73.91% |

**XGBoost**:

| Checkpoint | Accuracy | F1 | AUROC |
|------------|----------|-----|-------|
| Majority baseline | 70.10% | 82.42% | n/a |
| Probe (train) | 99.55% | 99.69% | 100.00% |
| Probe (val) | 74.81% | 84.55% | 77.02% |
| **Probe (test)** | **71.55%** | **82.68%** | **77.86%** |

**Why this worked:** logistic regression and MLP struggle with **high dimension (7194)** and **class imbalance** (483 hallucinated / 206 truthful). XGBoost with `scale_pos_weight` and nonlinear tree splits gave **+4.0 to 4.5 pp test AUROC**, the **largest gain** in the project

---

### Summary of metric gains

| Component | Test AUROC gain |
|-----------|----------------------------|
| Screening (8 heads) | move from impractical 43008 dim |
| Answer mask without EOS | ~+0.2 pp |
| Geometry (+26) | ~+0.02 pp |
| XGBoost | ~+4 pp |

---

## 3. Experiments not in the final solution

| Idea | Why I dropped it | Why it failed / was discarded |
|------|------------------|-------------------------------|
| All 48 pools without screening (43008 dim) | Too many features | With N=689 there are too many features per sample, so the probe overfits; screening is needed to compress the signal |
| Single random split instead of 5-fold | Less stable metrics | One split depends a lot on which 138 test examples land in the hold-out; 5-fold produces a more representative average |
| Geometry ablation by feature groups (MLP) | No AUROC gain per group | Subgroups add no new signal |

### XGBoost ablations (for choices from §2.1 - 2.3)

| Ablation | What I checked | Result with XGBoost | Why it failed / was discarded |
|----------|----------------|---------------------|-------------------------------|
| Mask: full seq. → answer only → answer without EOS | §2.2 | Same order as MLP: **answer without EOS** is best, gain about **~0.3 pp** vs full sequence | Confirms §2.2; the reasoning there still holds, but XGBoost did not make this effect much stronger |
| Pooling 7168 vs +26 geom. (7194) | §2.3 | Gain about **~0.1 pp** | Same as MLP: geometry barely moves AUROC; on this small dataset XGBoost also struggles to use these features |
| Geometry by subgroups (std / cosines / globals) | §2.3 detail | Gain about **~0.06 pp** | Larger angular drift between neighboring layer representations may correlate with hallucinations, but on this small dataset XGBoost also struggles to use these features|
