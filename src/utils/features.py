"""Shared feature engineering utilities for fraud-detection phases.

The Phase 01 baseline uses exactly three features so that later, richer models
can be measured against an intentionally minimal reference point.
"""

from typing import List

import pandas as pd


BASELINE_FEATURES: List[str] = [
    "TransactionAmt",
    "time_since_creation",
    "n_prior_transactions",
]


def prepare_baseline_features(df: pd.DataFrame, card_col: str = "card1") -> pd.DataFrame:
    """Build the 3-feature baseline table from raw IEEE-CIS transaction data.

    Adds (without mutating the input):

    - ``time_since_creation``: ``TransactionDT`` min-max normalized to ``[0, 1]``
      (used as a proxy for account age).
    - ``n_prior_transactions``: number of earlier transactions on the same card.

    ``TransactionAmt`` is assumed to already be present.

    Args:
        df: Transaction DataFrame.
        card_col: Column identifying the card/user.

    Returns:
        A copy of ``df`` with the baseline feature columns added.
    """
    out = df.copy()

    dt = out["TransactionDT"]
    dt_min = dt.min()
    dt_max = dt.max()
    if dt_max > dt_min:
        out["time_since_creation"] = (dt - dt_min) / (dt_max - dt_min)
    else:
        out["time_since_creation"] = 0.0

    out["n_prior_transactions"] = out.groupby(card_col).cumcount()

    return out
