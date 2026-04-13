"""Configuration management for CortexOS."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RecallWeights:
    """Weights for the recall scoring formula."""
    text_similarity: float = 0.4
    recency: float = 0.3
    zone_gravity: float = 0.2
    access_frequency: float = 0.1


@dataclass
class ZoneConfig:
    """Zone-related configuration."""
    emergence_threshold: int = 5
    dormant_days: int = 30
    archive_days: int = 90
    gravity_decay: float = 0.95
    merge_overlap_threshold: float = 0.7


@dataclass
class Config:
    """CortexOS configuration."""
    workspace: str = "./cortexos_data"
    agent_id: str = "default"
    recall_weights: RecallWeights = field(default_factory=RecallWeights)
    zone_config: ZoneConfig = field(default_factory=ZoneConfig)
    session_context_budget: int = 500
    recency_half_life_days: float = 7.0
    shared_zones: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create Config from a dictionary."""
        recall_weights = RecallWeights(**data.pop("recall_weights", {}))
        zone_config = ZoneConfig(**data.pop("zone_config", {}))
        return cls(recall_weights=recall_weights, zone_config=zone_config, **data)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from a YAML file."""
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def load(
        cls,
        workspace: Optional[str] = None,
        agent_id: Optional[str] = None,
        config: Any = None,
    ) -> "Config":
        """Smart config loader: accepts dict, YAML path, or defaults."""
        if isinstance(config, cls):
            cfg = config
        elif isinstance(config, dict):
            cfg = cls.from_dict(dict(config))
        elif isinstance(config, str) and os.path.isfile(config):
            cfg = cls.from_yaml(config)
        else:
            cfg = cls()

        if workspace is not None:
            cfg.workspace = workspace
        if agent_id is not None:
            cfg.agent_id = agent_id
        return cfg

    @property
    def memory_dir(self) -> Path:
        """Path to the memory storage directory."""
        return Path(self.workspace) / "memory"

    @property
    def zones_file(self) -> Path:
        """Path to zones metadata file."""
        return Path(self.workspace) / "memory" / "zones.yaml"

    @property
    def tasks_file(self) -> Path:
        """Path to tasks metadata file."""
        return Path(self.workspace) / "memory" / "tasks.yaml"
