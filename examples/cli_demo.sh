#!/bin/bash
# CLI Demo for Agent-CortexOS
set -e

WORKSPACE="/tmp/cortexos_demo"
echo "=== Agent-CortexOS CLI Demo ==="
echo "Workspace: $WORKSPACE"
echo

# Store memories
echo "--- Storing memories ---"
cortexos -w "$WORKSPACE" store "Kubernetes Pod restart: check OOMKilled, adjust memory limit" -t experience -e "K8s,Pod"
cortexos -w "$WORKSPACE" store "Docker multi-stage builds reduce image size by 60%" -t experience -e "Docker"
cortexos -w "$WORKSPACE" store "Meeting decision: migrate to K8s in Q2" -t decision -e "K8s"
cortexos -w "$WORKSPACE" store "PostgreSQL connection pooling improves throughput 3x" -t fact -e "PostgreSQL"
echo

# Create a zone
echo "--- Creating zone ---"
cortexos -w "$WORKSPACE" zones create devops -s "DevOps practices, CI/CD, infrastructure"
echo

# Recall memories
echo "--- Recalling: container issues ---"
cortexos -w "$WORKSPACE" recall "container memory issues"
echo

# List zones
echo "--- Zones ---"
cortexos -w "$WORKSPACE" zones list
echo

# Task management
echo "--- Creating tasks ---"
cortexos -w "$WORKSPACE" task create "Set up K8s monitoring" -p 2 -z devops
cortexos -w "$WORKSPACE" task create "Write deployment docs" -p 3
echo

echo "--- Task list ---"
cortexos -w "$WORKSPACE" task list
echo

# Stats
echo "--- System stats ---"
cortexos -w "$WORKSPACE" stats
echo

# Lifecycle operations
echo "--- Consolidation ---"
cortexos -w "$WORKSPACE" consolidate
echo

echo "=== Demo complete! ==="
echo "Inspect data: ls $WORKSPACE/memory/"
