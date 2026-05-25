"""Abstract base class for all data source collectors."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from soc_dsm.harmonizer.schema import CANONICAL_COLUMNS

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """
    All collectors inherit from this class and must implement:
      - download()  : fetch raw data from the remote source into raw_dir
      - parse()     : convert the downloaded files into the canonical schema
    """

    SOURCE_NAME: str = ""  # overridden in each subclass

    def __init__(self, raw_dir: str | Path):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(self, force_download: bool = False) -> pd.DataFrame:
        """Download (if needed) then parse into canonical schema."""
        if force_download or not self._is_downloaded():
            logger.info("[%s] Downloading raw data …", self.SOURCE_NAME)
            self.download()
        else:
            logger.info("[%s] Raw data already present, skipping download.", self.SOURCE_NAME)

        logger.info("[%s] Parsing …", self.SOURCE_NAME)
        df = self.parse()
        df = self._enforce_schema(df)
        logger.info("[%s] %d horizons loaded.", self.SOURCE_NAME, len(df))
        return df

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def download(self) -> None:
        """Fetch raw files into self.raw_dir."""

    @abstractmethod
    def parse(self) -> pd.DataFrame:
        """Return a DataFrame conforming to CANONICAL_COLUMNS."""

    @abstractmethod
    def _is_downloaded(self) -> bool:
        """Return True if raw data already exists on disk."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _enforce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add missing canonical columns as NaN and drop extras."""
        for col in CANONICAL_COLUMNS:
            if col not in df.columns:
                df[col] = float("nan")
        df["source_db"] = self.SOURCE_NAME
        return df[CANONICAL_COLUMNS]

    @staticmethod
    def _download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
        """Stream-download *url* to *dest*, show a progress bar."""
        import requests
        from tqdm import tqdm

        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in resp.iter_content(chunk_size):
                fh.write(chunk)
                bar.update(len(chunk))
        return dest
