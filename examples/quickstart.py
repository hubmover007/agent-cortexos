"""Quick start example — 5 lines to get going with CortexOS."""

import cortexos

# Initialize CortexOS (uses ./cortexos_data by default)
cx = cortexos.init()

# Store some memories
cx.store("Kubernetes Pod restart issue: check OOMKilled, adjust memory limit", 
         mem_type="experience", entities=["Kubernetes", "Pod", "OOMKilled"])
cx.store("Docker image optimization: multi-stage builds reduce image size by 60%",
         mem_type="experience", entities=["Docker"])
cx.store("Meeting decision: migrate to K8s in Q2",
         mem_type="decision", entities=["K8s"])

# Recall relevant memories
print("=== Recall: container memory problem ===")
results = cx.recall("container memory problem", budget=3)
for i, entry in enumerate(results, 1):
    print(f"  [{i}] ({entry.zone}) {entry.content}")

# Check system stats
print(f"\n=== Stats ===")
stats = cx.stats()
for k, v in stats.items():
    print(f"  {k}: {v}")

# Session context (for injecting into LLM system prompt)
print(f"\n=== Session Context ===")
context = cx.session_context(budget=300)
print(context)

# Save metadata
cx.save()
print("\nDone! Check ./cortexos_data/memory/ for stored data.")
