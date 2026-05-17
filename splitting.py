"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y,
    df=None,
    test_size=0.15,
    val_size=0.15,
    random_state=42,
):
    """Split dataset indices into train, validation, and test subsets.

    The default strategy performs a single stratified random split preserving
    the class ratio in each subset.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
                      Used for stratification.
        df:           Optional full DataFrame (same row order as ``y``).
                      Required for group-aware splits.
        test_size:    Fraction of samples reserved for the held-out test set.
        val_size:     Fraction of samples reserved for validation.
        random_state: Random seed for reproducible splits.

    Returns:
        A list of ``(idx_train, idx_val, idx_test)`` tuples of integer index
        arrays.  ``idx_val`` may be ``None``.

    Student task:
        Replace or extend the skeleton below.  The only contract is that the
        function returns the list described above.
    """

    idx = np.arange(len(y))

    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the split strategy below.
    # ------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    relative_val = val_size / 0.8
    splits = []
    for train_val_idx, idx_test in skf.split(idx, y):
        idx_train, idx_val = train_test_split(
            train_val_idx,
            test_size=relative_val,
            random_state=random_state,
            stratify=y[train_val_idx],
        )
        splits.append((idx_train, idx_val, idx_test))
    return splits
    # ------------------------------------------------------------------
