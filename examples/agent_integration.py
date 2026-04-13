"""Example: Integrating CortexOS with an AI Agent framework."""

import cortexos


def simulate_agent_session():
    """Simulate how an AI Agent would use CortexOS in a session."""
    
    # Initialize CortexOS for this agent
    cx = cortexos.init(workspace="./agent_workspace", agent_id="assistant-v1")

    # === Session Start: Inject Context ===
    system_context = cx.session_context(budget=500)
    print("=== Injected Session Context ===")
    print(system_context)
    print()

    # === During Conversation: Store Insights ===
    # When the agent learns something useful
    cx.store(
        "User prefers Python over JavaScript for backend services",
        mem_type="fact",
        entities=["Python", "JavaScript"],
    )

    # When a decision is made
    cx.store(
        "Decided to use FastAPI for the new microservice",
        mem_type="decision",
        entities=["FastAPI", "microservice"],
    )

    # === During Conversation: Recall When Needed ===
    print("=== Recalling: API framework ===")
    results = cx.recall("API framework recommendation")
    for r in results:
        print(f"  [{r.zone}] {r.content}")
    print()

    # === Task Management ===
    task = cx.tasks.create(
        "Set up FastAPI project structure",
        priority=2,
        due="2025-04-15T00:00:00+00:00",
    )
    cx.tasks.add_follow_up(
        task.id,
        "Review API endpoint design with user",
        due="2025-04-16T00:00:00+00:00",
    )

    print("=== Pending Tasks ===")
    for t in cx.tasks.list():
        print(f"  [{t.status.value}] {t.summary}")
        for fu in t.follow_ups:
            print(f"    → {fu.action} (due: {fu.due})")
    print()

    # === Session End: Consolidate ===
    result = cx.lifecycle.consolidate()
    print(f"Consolidation: {result}")

    # Save metadata
    cx.save()
    print("Session complete. Metadata saved.")


if __name__ == "__main__":
    simulate_agent_session()
