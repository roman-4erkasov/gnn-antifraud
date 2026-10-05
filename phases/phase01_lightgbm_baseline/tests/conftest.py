"""Shared pytest fixtures for the Phase 01 test suite."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PHASE_ROOT = Path(__file__).resolve().parents[1]
PHASE_SRC = PHASE_ROOT / "src"
PROJECT_ROOT = PHASE_ROOT.parents[1]

for _path in (str(PHASE_SRC), str(PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

FIXTURE_DATA_DIR = Path(__file__).resolve().parent / "data" / "ieee_cis_test"


@pytest.fixture
def fixture_data_dir() -> Path:
    """Path to the small synthetic IEEE-CIS fixture used by the e2e run."""
    return FIXTURE_DATA_DIR


def make_synthetic_transactions(n: int = 400, n_cards: int = 40, seed: int = 0) -> pd.DataFrame:
    """Build a small synthetic transaction frame with the baseline columns."""
    rng = np.random.RandomState(seed)
    cards = rng.randint(1, n_cards + 1, size=n)
    products = rng.choice(list("WCHSR"), size=n)
    amounts = rng.gamma(shape=2.0, scale=50.0, size=n)
    fraud = rng.binomial(1, 0.1, size=n)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionAmt": amounts,
            "TransactionDT": np.sort(rng.randint(0, 1_000_000, size=n)),
            "ProductCD": products,
            "card1": cards,
            "isFraud": fraud,
        }
    )


@pytest.fixture
def synthetic_transactions() -> pd.DataFrame:
    return make_synthetic_transactions()


@pytest.fixture
def synthetic_features(synthetic_transactions):
    """Return ``(X_train, y_train, X_val, y_val, X_test, y_test)`` arrays."""
    from src.utils.features import BASELINE_FEATURES, prepare_baseline_features

    df = prepare_baseline_features(synthetic_transactions)
    n = len(df)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    train, val, test = df.iloc[:n_train], df.iloc[n_train : n_train + n_val], df.iloc[n_train + n_val :]
    cols = BASELINE_FEATURES
    return (
        train[cols].to_numpy(dtype=float),
        train["isFraud"].to_numpy(dtype=int),
        val[cols].to_numpy(dtype=float),
        val["isFraud"].to_numpy(dtype=int),
        test[cols].to_numpy(dtype=float),
        test["isFraud"].to_numpy(dtype=int),
    )
