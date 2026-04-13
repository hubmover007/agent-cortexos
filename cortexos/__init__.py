"""Agent-CortexOS — Cognitive Operating System for AI Agents."""

from .client import CortexOS

__version__ = "0.1.0"


def init(workspace=None, agent_id=None, config=None):
    """Initialize a CortexOS instance.

    Args:
        workspace: Path to the memory workspace directory. Defaults to ./cortexos_data
        agent_id: Identifier for this agent. Defaults to 'default'
        config: Configuration dict or path to YAML config file.

    Returns:
        CortexOS instance ready to use.
    """
    return CortexOS(workspace=workspace, agent_id=agent_id, config=config)
