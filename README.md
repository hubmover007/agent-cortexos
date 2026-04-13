# Agent-CortexOS

> Cognitive Operating System for AI Agents — structured memory, cross-session state restoration, and proactive task management.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## The Problem

Every AI Agent today suffers from three fundamental flaws:

- **Memory is a timeline** — All information stacks linearly by time. Knowledge from last month mixes with today's conversation, making retrieval inefficient.
- **Every session starts from zero** — New session = amnesia. The agent forgets decisions, pitfalls, and accumulated experience.
- **Tasks don't self-advance** — Agents only respond to current instructions. They never proactively follow up on unfinished work.

## What is CortexOS?

```
CortexOS = Cognitive Substrate Layer for AI Agents

NOT: a vector database   NOT: chat history storage   NOT: an Agent framework
IS:  a working memory operating system for Agents
```

Three core capabilities:

- **Domain Memory** — Organize memories by semantic "gravity fields" (Zones). Knowledge auto-routes to the right domain and is retrievable across sessions.
- **State Restoration** — Inject full working context within 30 seconds of a new session. No need for users to re-explain background.
- **Task Driving** — Tasks automatically generate follow-up actions. Plug into a scheduler for proactive project advancement.

## Quick Start

### Install

```bash
pip install agent-cortexos
```

### 5 Lines of Code

```python
import cortexos

cx = cortexos.init()
cx.store("K8s Pod restart issue: check OOMKilled, adjust memory limit")
cx.store("Docker image optimization: multi-stage builds reduce image size by 60%")
results = cx.recall("container memory problem")
print(results[0].content)
```

### CLI

```bash
# Store a memory
cortexos store "Meeting decision: migrate to K8s in Q2" --type decision

# Recall memories
cortexos recall "Q2 plan"

# List zones
cortexos zones list

# System stats
cortexos stats

# Task management
cortexos task create "Finish API docs" --due 2025-04-30
cortexos task list
```

## Architecture

### Zone Routing (Three Layers)

Every memory entry is routed through three layers:

1. **Entity Reverse Lookup** — If the entry's entities match a Zone's known entities, route directly.
2. **TF-IDF Scope Match** — Compare entry content with Zone scope keywords using weighted similarity.
3. **Fallback** — Unmatched entries go to `_inbox`, where they may later trigger Zone emergence.

### Recall Scoring Formula

```
score = 0.4 × text_similarity + 0.3 × recency + 0.2 × zone_gravity + 0.1 × access_frequency
```

### Three-Layer Processing

| Layer | Latency | Operations |
|-------|---------|-----------|
| Hot (Real-time) | < 50ms | store, recall, session_context |
| Warm (Near-line) | seconds | zone discovery, dedup, scope updates |
| Cold (Offline) | minutes | gravity decay, archival, TF-IDF rebuild |

**Iron Rule: The main Agent never blocks.**

## API Reference

```python
import cortexos

cx = cortexos.init(workspace="./my_memory", agent_id="my-agent")

# Store
entry = cx.store("content", mem_type="experience", entities=["K8s", "Pod"])

# Recall
entries = cx.recall("query", budget=10, zones=["k8s-ops"])

# Session context (for system prompt injection)
context = cx.session_context(budget=500)

# Zone management
cx.zones.list()
cx.zones.create("devops", scope="DevOps practices")
cx.zones.stats("devops")
cx.zones.discover()  # Trigger emergence detection

# Task management
task = cx.tasks.create("Write docs", priority=2, due="2025-04-30")
cx.tasks.list()
cx.tasks.complete(task.id)
cx.tasks.add_follow_up(task.id, "Review with team", due="2025-05-01")

# Lifecycle
cx.lifecycle.consolidate()  # Warm path
cx.lifecycle.garden()        # Cold path

# Stats
cx.stats()

# Persist metadata
cx.save()
```

## Multi-Agent Support

Three modes for multi-agent scenarios:

- **shared** — All agents share the same workspace and memories.
- **isolated** — Each agent has a completely separate workspace.
- **selective** — Private zones + subscribed shared zones.

## Roadmap

- **v0.1** (current) — JSONL storage, TF-IDF retrieval, Zone routing, Task system, CLI
- **v0.2** — Vector embeddings, hybrid retrieval
- **v0.3** — Multi-agent collaboration
- **v0.5** — SQLite/PostgreSQL backends
- **v1.0** — Production-ready with REST API, Docker, plugin system

## Contributing

```bash
git clone https://github.com/anthropic/agent-cortexos.git
cd agent-cortexos
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

See [CONTRIBUTING](docs/design.md) for detailed guidelines.

## License

MIT — see [LICENSE](LICENSE).
