"""Shared feature engineering utilities for fraud-detection phases.

The Phase 01 baseline uses exactly three features so that later, richer models
can be measured against an intentionally minimal reference point.

The API is split into a *fit* and a *transform* step so that the fitted
parameters (currently the ``TransactionDT`` min/max) are learned from the
training split only and then applied to validation/test, avoiding leakage.
"""

from typing import Dict, List, Optional

import pandas as pd


BASELINE_FEATURES: List[str] = [
    "TransactionAmt",
    "time_since_creation",
    "n_prior_transactions",
]


def fit_baseline_feature_params(
    df: pd.DataFrame,
    card_col: str = "card1",
) -> Dict[str, float]:
    """Fit the parameters needed to build the baseline features.

    Must be called on the *training* split only. The returned parameters are
    then passed to :func:`prepare_baseline_features` for every split.

    Args:
        df: Training transaction DataFrame.
        card_col: Column identifying the card/user (kept for API symmetry).

    Returns:
        Dict with the fitted ``dt_min`` and ``dt_max`` values.
    """
    dt = df["TransactionDT"]
    return {
        "dt_min": float(dt.min()),
        "dt_max": float(dt.max()),
    }


def prepare_baseline_features(
    df: pd.DataFrame,
    params: Optional[Dict[str, float]] = None,
    card_col: str = "card1",
) -> pd.DataFrame:
    """Build the 3-feature baseline table from raw IEEE-CIS transaction data.

    Adds (without mutating the input):

    - ``time_since_creation``: ``TransactionDT`` min-max normalized to ``[0, 1]``
      using ``params`` (used as a proxy for account age).
    - ``n_prior_transactions``: number of earlier transactions on the same card,
      counted within ``df`` only.

    ``TransactionAmt`` is assumed to already be present.

    When ``params`` is ``None`` the normalization parameters are fitted on
    ``df`` itself, which is convenient for standalone use (e.g. lessons) but
    leaks information if ``df`` is not the training split. Callers that split
    data should fit on the training split via :func:`fit_baseline_feature_params`
    and pass the result here.

    Args:
        df: Transaction DataFrame.
        params: Fitted parameters from :func:`fit_baseline_feature_params`.
        card_col: Column identifying the card/user.

    Returns:
        A copy of ``df`` with the baseline feature columns added.
    """
    out = df.copy()

    if params is None:
        params = fit_baseline_feature_params(df, card_col=card_col)

    dt_min = params["dt_min"]
    dt_max = params["dt_max"]
    dt = out["TransactionDT"]
    if dt_max > dt_min:
        out["time_since_creation"] = ((dt - dt_min) / (dt_max - dt_min)).clip(0.0, 1.0)
    else:
        out["time_since_creation"] = 0.0

    out["n_prior_transactions"] = out.groupby(card_col).cumcount()

    return out
