"""CortexOS CLI — Command-line interface powered by Click."""

from __future__ import annotations

import json
import sys

import click

import cortexos


@click.group()
@click.option("--workspace", "-w", default=None, help="Path to workspace directory.")
@click.option("--agent-id", "-a", default=None, help="Agent identifier.")
@click.pass_context
def cli(ctx, workspace, agent_id):
    """Agent-CortexOS — Cognitive Operating System for AI Agents."""
    ctx.ensure_object(dict)
    ctx.obj["workspace"] = workspace
    ctx.obj["agent_id"] = agent_id


def _get_cx(ctx) -> cortexos.CortexOS:
    """Lazy-init CortexOS from context."""
    return cortexos.init(
        workspace=ctx.obj.get("workspace"),
        agent_id=ctx.obj.get("agent_id"),
    )


@cli.command()
@click.argument("content")
@click.option("--type", "-t", "mem_type", default="note", help="Memory type.")
@click.option("--zone", "-z", default=None, help="Explicit zone assignment.")
@click.option("--entities", "-e", default=None, help="Comma-separated entities.")
@click.pass_context
def store(ctx, content, mem_type, zone, entities):
    """Store a new memory entry."""
    cx = _get_cx(ctx)
    entity_list = [e.strip() for e in entities.split(",")] if entities else None
    entry = cx.store(content, mem_type=mem_type, zone=zone, entities=entity_list)
    cx.save()
    click.echo(f"Stored: [{entry.zone}] {entry.id[:8]}... ({entry.mem_type})")


@cli.command()
@click.argument("query")
@click.option("--budget", "-n", default=5, help="Max results to return.")
@click.option("--zones", "-z", default=None, help="Comma-separated zone filter.")
@click.option("--types", "-t", default=None, help="Comma-separated type filter.")
@click.pass_context
def recall(ctx, query, budget, zones, types):
    """Recall relevant memories."""
    cx = _get_cx(ctx)
    zone_list = [z.strip() for z in zones.split(",")] if zones else None
    type_list = [t.strip() for t in types.split(",")] if types else None
    entries = cx.recall(query, budget=budget, zones=zone_list, mem_types=type_list)

    if not entries:
        click.echo("No relevant memories found.")
        return

    for i, entry in enumerate(entries, 1):
        click.echo(f"\n--- [{i}] Zone: {entry.zone} | Type: {entry.mem_type} ---")
        click.echo(entry.content)


@cli.group()
def zones():
    """Manage knowledge zones."""
    pass


@zones.command("list")
@click.option("--all", "show_all", is_flag=True, help="Include dormant zones.")
@click.pass_context
def zones_list(ctx, show_all):
    """List all zones."""
    cx = _get_cx(ctx)
    zone_list = cx.zones.list(include_dormant=show_all)

    if not zone_list:
        click.echo("No zones yet. Store some memories to trigger zone emergence.")
        return

    for z in zone_list:
        status_icon = {"active": "●", "dormant": "○", "archived": "◌"}.get(
            z.status.value, "?"
        )
        click.echo(
            f"  {status_icon} {z.name:20s} gravity={z.gravity:.1f}  "
            f"entries={z.entry_count}  [{z.scope[:40]}]"
        )


@zones.command("create")
@click.argument("name")
@click.option("--scope", "-s", default="", help="Zone scope description.")
@click.pass_context
def zones_create(ctx, name, scope):
    """Create a zone manually."""
    cx = _get_cx(ctx)
    zone = cx.zones.create(name=name, scope=scope)
    cx.save()
    click.echo(f"Created zone: {zone.name}")


@zones.command("stats")
@click.argument("name")
@click.pass_context
def zones_stats(ctx, name):
    """Show zone statistics."""
    cx = _get_cx(ctx)
    try:
        s = cx.zones.stats(name)
        for k, v in s.items():
            click.echo(f"  {k}: {v}")
    except ValueError as e:
        click.echo(str(e), err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def stats(ctx):
    """Show overall system statistics."""
    cx = _get_cx(ctx)
    s = cx.stats()
    click.echo("CortexOS Statistics:")
    for k, v in s.items():
        click.echo(f"  {k}: {v}")


@cli.group()
def task():
    """Manage tasks."""
    pass


@task.command("create")
@click.argument("summary")
@click.option("--priority", "-p", default=3, type=int, help="Priority (1=high, 5=low).")
@click.option("--due", "-d", default=None, help="Due date (ISO 8601).")
@click.option("--zone", "-z", default=None, help="Related zone.")
@click.pass_context
def task_create(ctx, summary, priority, due, zone):
    """Create a new task."""
    cx = _get_cx(ctx)
    t = cx.tasks.create(summary=summary, priority=priority, due=due, zone=zone)
    cx.save()
    click.echo(f"Created task: {t.id[:8]}... '{t.summary}'")


@task.command("list")
@click.option("--status", "-s", default=None, help="Filter by status.")
@click.pass_context
def task_list(ctx, status):
    """List tasks."""
    cx = _get_cx(ctx)
    from cortexos.models.task import TaskStatus

    status_filter = TaskStatus(status) if status else None
    tasks = cx.tasks.list(status=status_filter)

    if not tasks:
        click.echo("No tasks found.")
        return

    for t in tasks:
        icon = {"todo": "☐", "doing": "◑", "done": "✓", "blocked": "✗", "cancelled": "–"}.get(
            t.status.value, "?"
        )
        due_str = f" due:{t.due}" if t.due else ""
        click.echo(f"  {icon} [{t.priority}] {t.summary}{due_str}  ({t.id[:8]})")


@task.command("complete")
@click.argument("task_id")
@click.pass_context
def task_complete(ctx, task_id):
    """Mark a task as completed."""
    cx = _get_cx(ctx)
    # Support partial ID matching
    matching = [tid for tid in cx.tasks._tasks if tid.startswith(task_id)]
    if len(matching) == 0:
        click.echo(f"No task found matching '{task_id}'", err=True)
        sys.exit(1)
    if len(matching) > 1:
        click.echo(f"Ambiguous ID '{task_id}', matches: {matching}", err=True)
        sys.exit(1)

    t = cx.tasks.complete(matching[0])
    cx.save()
    click.echo(f"Completed: '{t.summary}'")


@cli.command()
@click.pass_context
def consolidate(ctx):
    """Run warm-path lifecycle operations."""
    cx = _get_cx(ctx)
    result = cx.lifecycle.consolidate()
    cx.save()
    click.echo("Consolidation complete:")
    for k, v in result.items():
        click.echo(f"  {k}: {v}")


@cli.command()
@click.pass_context
def garden(ctx):
    """Run cold-path lifecycle operations."""
    cx = _get_cx(ctx)
    result = cx.lifecycle.garden()
    cx.save()
    click.echo("Garden complete:")
    for k, v in result.items():
        click.echo(f"  {k}: {v}")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
