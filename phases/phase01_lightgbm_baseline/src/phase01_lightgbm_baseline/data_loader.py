"""Data loading utilities for the IEEE-CIS fraud detection dataset."""

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


REQUIRED_FILES: List[str] = [
    "train_transaction.csv",
    "test_transaction.csv",
    "train_identity.csv",
    "test_identity.csv",
]


class DataLoader:
    """Load the IEEE-CIS dataset from the shared data directory.

    The loader never fabricates data: it validates that the required CSVs are
    present and, when they are not, prints instructions for obtaining them.
    """

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)
        self.ieee_cis_dir = self._resolve_dir()

    def _resolve_dir(self) -> Path:
        """Resolve the directory holding the CSVs.

        Accepts either a directory that directly contains the four CSV files
        (e.g. a test fixture) or a parent directory with an ``ieee-cis/``
        subdirectory (the shared ``data/`` layout).
        """
        if all((self.data_dir / f).exists() for f in REQUIRED_FILES):
            return self.data_dir
        return self.data_dir / "ieee-cis"

    def check_data_exists(self) -> bool:
        """Return True if every required IEEE-CIS CSV is present."""
        self.ieee_cis_dir = self._resolve_dir()
        if not self.ieee_cis_dir.exists():
            return False
        return all((self.ieee_cis_dir / f).exists() for f in REQUIRED_FILES)

    def download_data(self) -> None:
        """Validate the dataset and raise with download instructions if missing.

        IEEE-CIS is distributed on Kaggle and requires accepting its terms, so
        it cannot be downloaded automatically. This method reports what is
        present/missing and how to obtain the data.
        """
        self.ieee_cis_dir = self._resolve_dir()
        self.ieee_cis_dir.mkdir(parents=True, exist_ok=True)

        present = [f for f in REQUIRED_FILES if (self.ieee_cis_dir / f).exists()]
        missing = [f for f in REQUIRED_FILES if f not in present]

        print(f"IEEE-CIS data directory: {self.ieee_cis_dir}")
        print(f"  present: {present or 'none'}")
        print(f"  missing: {missing or 'none'}")

        if not missing:
            return

        print()
        print("To obtain the IEEE-CIS Fraud Detection dataset:")
        print("  1. Accept the competition rules at https://www.kaggle.com/c/ieee-fraud-detection/data")
        print("  2. Download ieee-fraud-detection.zip (Kaggle CLI: `kaggle competitions download -c ieee-fraud-detection`)")
        print(f"  3. Unzip the four CSVs into: {self.ieee_cis_dir}")
        raise FileNotFoundError(
            f"Missing IEEE-CIS files in {self.ieee_cis_dir}: {missing}"
        )

    def load_data(
        self,
        sample_limit: Optional[int] = None,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train/test DataFrames with raw merged columns.

        The frames contain the raw transaction/identity columns only. Baseline
        features are prepared later by the pipeline, after the train/validation
        split, so that fitted parameters never see validation/test rows.

        Args:
            sample_limit: If provided, take a stratified subsample of this size.
            random_state: Seed used for the stratified subsample.

        Returns:
            Tuple of ``(train_df, test_df)``.
        """
        if not self.check_data_exists():
            self.download_data()

        train_txn = pd.read_csv(self.ieee_cis_dir / "train_transaction.csv")
        test_txn = pd.read_csv(self.ieee_cis_dir / "test_transaction.csv")
        train_id = pd.read_csv(self.ieee_cis_dir / "train_identity.csv")
        test_id = pd.read_csv(self.ieee_cis_dir / "test_identity.csv")

        train_df = train_txn.merge(train_id, on="TransactionID", how="left")
        test_df = test_txn.merge(test_id, on="TransactionID", how="left")

        if sample_limit is not None:
            train_df = self._stratified_sample(train_df, sample_limit, random_state)
            test_df = self._stratified_sample(test_df, sample_limit, random_state)

        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    @staticmethod
    def _stratified_sample(
        df: pd.DataFrame,
        n: int,
        random_state: int = 42,
        label_col: str = "isFraud",
    ) -> pd.DataFrame:
        """Return a stratified subsample of at most ``n`` rows."""
        if n is None or n >= len(df):
            return df
        if label_col not in df.columns or df[label_col].nunique() < 2:
            return df.sample(n=n, random_state=random_state).reset_index(drop=True)

        frac = n / len(df)
        parts = []
        for _, group in df.groupby(label_col):
            take = max(1, int(round(frac * len(group))))
            take = min(take, len(group))
            parts.append(group.sample(n=take, random_state=random_state))
        sampled = pd.concat(parts)
        return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
