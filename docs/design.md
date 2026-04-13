# Agent-CortexOS Design Document

## Overview

Agent-CortexOS is a Cognitive Operating System for AI Agents. It provides structured memory management, cross-session state restoration, and proactive task management capabilities.

## Design Principles

1. **Agent-First**: Designed for AI Agent workflows, not human note-taking
2. **Zero-Config Start**: Works out of the box with sensible defaults
3. **Non-Blocking**: The main Agent path (store/recall) is always fast (< 50ms)
4. **Self-Organizing**: Zones emerge automatically from patterns in stored memories
5. **Pluggable Storage**: Default JSONL backend, extensible to SQL/vector stores

## Architecture

### Core Data Flow

```
Agent Interaction
      │
      ├── store(content) ──→ Entity Extract ──→ Zone Router ──→ JSONL Write + Index Update
      │
      └── recall(query) ──→ TF-IDF Search ──→ Multi-Factor Score ──→ Return Top-K
```

### Zone Routing — Three Layers

1. **Entity Reverse Lookup**: O(1) dictionary lookup. If entry entities exist in a zone's entity set, route there.
2. **TF-IDF Scope Match**: Compare entry token set with zone keyword sets. Weighted by zone gravity.
3. **Fallback to _inbox**: Unmatched entries accumulate in _inbox. When a cluster of similar entries reaches the emergence threshold (default: 5), a new Zone is automatically created.

### Recall Scoring

The recall engine uses a multi-factor scoring formula:

```
score = w1 * text_sim + w2 * recency + w3 * gravity + w4 * freq
```

Where:
- `text_sim` (0.4): TF-IDF cosine similarity between query and entry
- `recency` (0.3): Exponential decay with configurable half-life (default: 7 days)
- `gravity` (0.2): Zone gravity normalized to [0, 1]
- `freq` (0.1): Access count normalized to [0, 1]

### Storage

JSONL (JSON Lines) format, sharded by month:

```
memory/
  2025-03.jsonl
  2025-04.jsonl
  zones.yaml
  tasks.yaml
```

Benefits:
- Append-only = no write conflicts
- Human-readable and debuggable
- Monthly sharding = predictable file sizes
- Easy backup and migration

### Multi-Agent Modes

- **shared**: Single workspace, all agents see everything
- **isolated**: Separate workspaces per agent
- **selective**: Private zones + shared zone subscriptions

## Technology Choices

- **Python 3.9+**: Maximum compatibility
- **dataclass + typing**: Clean models, IDE-friendly
- **JSONL**: Simple, debuggable, no external dependencies for storage
- **TF-IDF**: No external API needed for retrieval (v0.2 will add optional vector support)
- **Click**: Clean CLI framework
- **PyYAML**: Configuration and metadata

## Future Considerations

- Vector embedding support (OpenAI, local models)
- SQLite/PostgreSQL backends
- REST API server mode
- Plugin system for custom storage/retrieval backends
- WebSocket real-time sync for multi-agent scenarios
