"""JSONL storage backend — append-only, monthly-sharded files."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

from ..models.entry import Entry


class JSONLBackend:
    """Append-only JSONL storage with monthly sharding.

    Files are stored as: <memory_dir>/YYYY-MM.jsonl
    Each line is a JSON object representing one Entry.
    """

    def __init__(self, memory_dir: str | Path):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def _shard_path(self, ts: str) -> Path:
        """Get the shard file path for a given timestamp."""
        try:
            dt = datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            dt = datetime.now(timezone.utc)
        return self.memory_dir / f"{dt.strftime('%Y-%m')}.jsonl"

    def append(self, entry: Entry) -> None:
        """Append an entry to the appropriate monthly shard."""
        path = self._shard_path(entry.ts)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def scan(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Iterator[Entry]:
        """Scan entries across shards within an optional time range.

        Args:
            start: ISO 8601 start time (inclusive). None = no lower bound.
            end: ISO 8601 end time (exclusive). None = no upper bound.

        Yields:
            Entry objects in chronological order.
        """
        shard_files = sorted(self.memory_dir.glob("*.jsonl"))

        for shard_file in shard_files:
            # Quick shard-level filtering by filename (YYYY-MM)
            shard_month = shard_file.stem  # e.g., "2025-04"
            if start and shard_month < start[:7]:
                continue
            if end and shard_month > end[:7]:
                continue

            with open(shard_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        entry = Entry.from_dict(data)
                    except (json.JSONDecodeError, KeyError):
                        continue

                    # Fine-grained time filtering
                    if start and entry.ts < start:
                        continue
                    if end and entry.ts >= end:
                        continue
                    yield entry

    def load_all(self) -> List[Entry]:
        """Load all entries from all shards into memory."""
        return list(self.scan())

    def count(self) -> int:
        """Count total entries across all shards."""
        total = 0
        for shard_file in self.memory_dir.glob("*.jsonl"):
            with open(shard_file, "r", encoding="utf-8") as f:
                total += sum(1 for line in f if line.strip())
        return total

    def update_entry(self, entry: Entry) -> None:
        """Update an entry in-place (rewrite the shard).

        This is an expensive operation — only used for updating access_count etc.
        In practice, we batch these in consolidate().
        """
        path = self._shard_path(entry.ts)
        if not path.exists():
            return

        lines = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("id") == entry.id:
                        lines.append(json.dumps(entry.to_dict(), ensure_ascii=False))
                    else:
                        lines.append(line)
                except json.JSONDecodeError:
                    lines.append(line)

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n" if lines else "")
