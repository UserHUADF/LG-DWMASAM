"""Input helpers for LG-DWMASAM experiments."""

from __future__ import annotations

import csv
from pathlib import Path

from .core import TemporalNetwork


REQUIRED_COLUMNS = ("time", "source", "target", "weight")


def read_temporal_csv(path: str | Path) -> TemporalNetwork:
    """Read a temporal edge CSV with columns: time, source, target, weight."""

    path = Path(path)
    records: list[tuple[str, str, str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            try:
                time = str(row["time"]).strip()
                source = str(row["source"]).strip()
                target = str(row["target"]).strip()
                weight = float(row["weight"])
            except Exception as exc:  # pragma: no cover - message is what matters
                raise ValueError(f"invalid CSV row {row_number}: {row}") from exc
            if not time or not source or not target:
                raise ValueError(f"blank time/source/target in row {row_number}")
            records.append((time, source, target, weight))

    return TemporalNetwork.from_records(records)
