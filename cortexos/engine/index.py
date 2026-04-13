"""In-memory index for fast retrieval — inverted, entity, zone, and temporal indexes."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set

from ..models.entry import Entry


def tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric, filter short tokens."""
    tokens = re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", text.lower())
    return [t for t in tokens if len(t) > 1]


class MemoryIndex:
    """In-memory index built from entries for fast retrieval.

    Maintains four index structures:
    - inverted: token -> set of entry IDs (for TF-IDF)
    - entity: entity -> set of entry IDs (for entity lookup)
    - zone: zone_name -> set of entry IDs
    - entries: id -> Entry (master store)
    """

    def __init__(self):
        self.entries: Dict[str, Entry] = {}
        self.inverted: Dict[str, Set[str]] = defaultdict(set)
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)
        self.zone_index: Dict[str, Set[str]] = defaultdict(set)
        # Document frequency: token -> count of docs containing it
        self._doc_freq: Dict[str, int] = defaultdict(int)
        # Per-doc token frequency
        self._term_freq: Dict[str, Dict[str, int]] = {}

    def add(self, entry: Entry) -> None:
        """Add an entry to all indexes."""
        self.entries[entry.id] = entry

        # Token index + TF-IDF stats
        tokens = tokenize(entry.content)
        seen_tokens: Set[str] = set()
        tf: Dict[str, int] = defaultdict(int)

        for token in tokens:
            self.inverted[token].add(entry.id)
            tf[token] += 1
            if token not in seen_tokens:
                self._doc_freq[token] += 1
                seen_tokens.add(token)

        self._term_freq[entry.id] = dict(tf)

        # Entity index
        for entity in entry.entities:
            self.entity_index[entity.lower()].add(entry.id)

        # Zone index
        self.zone_index[entry.zone].add(entry.id)

    def remove(self, entry_id: str) -> None:
        """Remove an entry from all indexes."""
        entry = self.entries.pop(entry_id, None)
        if entry is None:
            return

        tokens = tokenize(entry.content)
        seen: Set[str] = set()
        for token in tokens:
            self.inverted[token].discard(entry_id)
            if token not in seen:
                self._doc_freq[token] = max(0, self._doc_freq.get(token, 0) - 1)
                seen.add(token)

        self._term_freq.pop(entry_id, None)

        for entity in entry.entities:
            self.entity_index[entity.lower()].discard(entry_id)

        self.zone_index.get(entry.zone, set()).discard(entry_id)

    def build(self, entries: List[Entry]) -> None:
        """Rebuild index from scratch."""
        self.entries.clear()
        self.inverted.clear()
        self.entity_index.clear()
        self.zone_index.clear()
        self._doc_freq.clear()
        self._term_freq.clear()
        for entry in entries:
            self.add(entry)

    def tfidf_search(
        self,
        query: str,
        zones: Optional[List[str]] = None,
        mem_types: Optional[List[str]] = None,
    ) -> List[tuple]:
        """TF-IDF cosine similarity search.

        Args:
            query: Search query text.
            zones: Optional list of zones to restrict search to.
            mem_types: Optional list of memory types to filter.

        Returns:
            List of (entry_id, score) sorted by descending score.
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        n_docs = max(len(self.entries), 1)

        # Compute query TF-IDF vector
        query_tf: Dict[str, int] = defaultdict(int)
        for t in query_tokens:
            query_tf[t] += 1

        query_vec: Dict[str, float] = {}
        for t, freq in query_tf.items():
            df = self._doc_freq.get(t, 0)
            if df > 0:
                idf = math.log(n_docs / df)
                query_vec[t] = freq * idf

        if not query_vec:
            return []

        # Candidate entries (union of posting lists)
        candidates: Set[str] = set()
        for t in query_vec:
            candidates |= self.inverted.get(t, set())

        # Filter by zones and mem_types
        if zones:
            zone_entries: Set[str] = set()
            for z in zones:
                zone_entries |= self.zone_index.get(z, set())
            candidates &= zone_entries

        if mem_types:
            candidates = {
                eid for eid in candidates
                if self.entries[eid].mem_type in mem_types
            }

        # Score each candidate
        query_norm = math.sqrt(sum(v * v for v in query_vec.values()))
        results = []

        for eid in candidates:
            doc_tf = self._term_freq.get(eid, {})
            dot = 0.0
            doc_norm_sq = 0.0
            for t, q_weight in query_vec.items():
                d_freq = doc_tf.get(t, 0)
                if d_freq > 0:
                    df = self._doc_freq.get(t, 1)
                    d_weight = d_freq * math.log(n_docs / df)
                    dot += q_weight * d_weight
                    doc_norm_sq += d_weight * d_weight

            if dot > 0 and doc_norm_sq > 0:
                score = dot / (query_norm * math.sqrt(doc_norm_sq))
                results.append((eid, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def entity_lookup(self, entities: List[str]) -> Set[str]:
        """Find entries that mention any of the given entities."""
        result: Set[str] = set()
        for entity in entities:
            result |= self.entity_index.get(entity.lower(), set())
        return result

    def zone_entries(self, zone_name: str) -> Set[str]:
        """Get all entry IDs in a given zone."""
        return self.zone_index.get(zone_name, set())

    @property
    def total_entries(self) -> int:
        return len(self.entries)
