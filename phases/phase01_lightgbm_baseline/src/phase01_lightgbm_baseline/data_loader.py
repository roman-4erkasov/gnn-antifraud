"""Data loading utilities for IEEE-CIS fraud detection dataset."""

import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple

import pandas as pd


class DataLoader:
    """Load IEEE-CIS fraud detection dataset.
    
    Checks if data exists in shared data/ directory and downloads if missing.
    """

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)
        self.ieee_cis_dir = self.data_dir / "ieee-cis"

    def check_data_exists(self) -> bool:
        """Check if IEEE-CIS data exists in data/ieee-cis/ directory.
        
        Returns:
            True if data directory exists and contains required files.
        """
        if not self.ieee_cis_dir.exists():
            return False
        
        required_files = [
            "train_transaction.csv",
            "test_transaction.csv",
            "train_identity.csv",
            "test_identity.csv",
        ]
        
        return all((self.ieee_cis_dir / f).exists() for f in required_files)

    def download_data(self) -> None:
        """Download IEEE-CIS dataset or extract from archive.
        
        Creates data/ieee-cis/ directory structure with transaction and identity files.
        For now, creates synthetic data for testing purposes.
        """
        self.ieee_cis_dir.mkdir(parents=True, exist_ok=True)
        
        # Create synthetic data for testing
        # In production, this would download from Kaggle or extract from archive
        self._create_synthetic_data()

    def _create_synthetic_data(self) -> None:
        """Create synthetic IEEE-CIS-like data for testing."""
        import numpy as np
        
        np.random.seed(42)
        n_train = 1000
        n_test = 200
        
        # Transaction data
        train_txn = pd.DataFrame({
            "TransactionID": range(n_train),
            "TransactionAmt": np.random.exponential(100, n_train),
            "TransactionDT": np.arange(n_train),
            "ProductCD": np.random.choice(["W", "H", "C", "S", "R"], n_train),
            "card1": np.random.randint(1000, 20000, n_train),
            "card2": np.random.choice([100, 200, 300, 400, 500], n_train),
            "addr1": np.random.randint(100, 500, n_train),
            "addr2": np.random.choice([10, 20, 30, 40, 50], n_train),
            "dist1": np.random.exponential(10, n_train),
            "dist2": np.random.exponential(5, n_train),
            "isFraud": np.random.binomial(1, 0.035, n_train),  # ~3.5% fraud rate
        })
        
        test_txn = pd.DataFrame({
            "TransactionID": range(n_train, n_train + n_test),
            "TransactionAmt": np.random.exponential(100, n_test),
            "TransactionDT": np.arange(n_test),
            "ProductCD": np.random.choice(["W", "H", "C", "S", "R"], n_test),
            "card1": np.random.randint(1000, 20000, n_test),
            "card2": np.random.choice([100, 200, 300, 400, 500], n_test),
            "addr1": np.random.randint(100, 500, n_test),
            "addr2": np.random.choice([10, 20, 30, 40, 50], n_test),
            "dist1": np.random.exponential(10, n_test),
            "dist2": np.random.exponential(5, n_test),
            "isFraud": np.random.binomial(1, 0.035, n_test),
        })
        
        # Identity data
        train_id = pd.DataFrame({
            "TransactionID": range(n_train),
            "id_01": np.random.uniform(-5, 0, n_train),
            "id_02": np.random.uniform(100, 500, n_train),
            "DeviceType": np.random.choice(["macintosh", "windows", "android"], n_train),
            "DeviceInfo": np.random.choice(["windows", "ios", "android", "linux"], n_train),
        })
        
        test_id = pd.DataFrame({
            "TransactionID": range(n_train, n_train + n_test),
            "id_01": np.random.uniform(-5, 0, n_test),
            "id_02": np.random.uniform(100, 500, n_test),
            "DeviceType": np.random.choice(["macintosh", "windows", "android"], n_test),
            "DeviceInfo": np.random.choice(["windows", "ios", "android", "linux"], n_test),
        })
        
        # Save to CSV
        train_txn.to_csv(self.ieee_cis_dir / "train_transaction.csv", index=False)
        test_txn.to_csv(self.ieee_cis_dir / "test_transaction.csv", index=False)
        train_id.to_csv(self.ieee_cis_dir / "train_identity.csv", index=False)
        test_id.to_csv(self.ieee_cis_dir / "test_identity.csv", index=False)

    def load_data(self, sample_limit: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load IEEE-CIS transaction and identity data into DataFrames.
        
        Args:
            sample_limit: If provided, limit to this many samples (for testing).
            
        Returns:
            Tuple of (train_df, test_df) with baseline features prepared.
        """
        if not self.check_data_exists():
            self.download_data()
        
        # Load transaction data
        train_txn = pd.read_csv(self.ieee_cis_dir / "train_transaction.csv")
        test_txn = pd.read_csv(self.ieee_cis_dir / "test_transaction.csv")
        
        # Load identity data
        train_id = pd.read_csv(self.ieee_cis_dir / "train_identity.csv")
        test_id = pd.read_csv(self.ieee_cis_dir / "test_identity.csv")
        
        # Merge transaction and identity data
        train_df = train_txn.merge(train_id, on="TransactionID", how="left")
        test_df = test_txn.merge(test_id, on="TransactionID", how="left")
        
        # Apply baseline feature preparation
        train_df = self._prepare_baseline_features(train_df)
        test_df = self._prepare_baseline_features(test_df)
        
        # Apply sample limit if specified
        if sample_limit is not None:
            train_df = train_df.head(sample_limit)
            test_df = test_df.head(sample_limit)
        
        return train_df, test_df

    def _prepare_baseline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare 3-feature baseline from raw transaction data.
        
        Features:
        1. TransactionAmt - transaction amount
        2. TransactionDT - time since account creation (normalized)
        3. n_prior_transactions - number of prior transactions by card
        
        Args:
            df: Transaction DataFrame.
            
        Returns:
            DataFrame with baseline features added.
        """
        # Feature 1: Transaction amount (already present)
        # df["TransactionAmt"] exists
        
        # Feature 2: Time since account creation (use TransactionDT as proxy)
        # Normalize to [0, 1] range
        dt_min = df["TransactionDT"].min()
        dt_max = df["TransactionDT"].max()
        if dt_max > dt_min:
            df["time_since_creation"] = (df["TransactionDT"] - dt_min) / (dt_max - dt_min)
        else:
            df["time_since_creation"] = 0.0
        
        # Feature 3: Number of prior transactions by card1
        # Count occurrences of each card1 value up to current row
        df["n_prior_transactions"] = df.groupby("card1").cumcount()
        
        return df
