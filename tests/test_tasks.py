"""Tests for Task management system."""

import shutil
import tempfile

import pytest

import cortexos
from cortexos.models.task import TaskStatus


@pytest.fixture
def workspace():
    tmpdir = tempfile.mkdtemp(prefix="cortexos_test_tasks_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def cx(workspace):
    return cortexos.init(workspace=workspace)


class TestTaskCRUD:
    def test_create_task(self, cx):
        task = cx.tasks.create("Write documentation")
        assert task.summary == "Write documentation"
        assert task.status == TaskStatus.TODO
        assert task.priority == 3

    def test_create_task_with_options(self, cx):
        task = cx.tasks.create(
            "Urgent fix",
            priority=1,
            due="2025-04-30T00:00:00+00:00",
            zone="ops",
        )
        assert task.priority == 1
        assert task.due == "2025-04-30T00:00:00+00:00"
        assert task.zone == "ops"

    def test_list_tasks(self, cx):
        cx.tasks.create("Task A", priority=2)
        cx.tasks.create("Task B", priority=1)
        cx.tasks.create("Task C", priority=3)

        tasks = cx.tasks.list()
        assert len(tasks) == 3
        # Sorted by priority
        assert tasks[0].priority <= tasks[1].priority

    def test_list_filter_by_status(self, cx):
        t1 = cx.tasks.create("Active task")
        t2 = cx.tasks.create("Done task")
        cx.tasks.complete(t2.id)

        active = cx.tasks.list(status=TaskStatus.TODO)
        assert len(active) == 1
        assert active[0].id == t1.id

    def test_update_task(self, cx):
        task = cx.tasks.create("Original summary")
        cx.tasks.update(task.id, summary="Updated summary", priority=1)

        updated = cx.tasks.get(task.id)
        assert updated.summary == "Updated summary"
        assert updated.priority == 1

    def test_update_nonexistent_raises(self, cx):
        with pytest.raises(ValueError, match="not found"):
            cx.tasks.update("nonexistent-id", summary="test")

    def test_complete_task(self, cx):
        task = cx.tasks.create("Completable task")
        completed = cx.tasks.complete(task.id)
        assert completed.status == TaskStatus.DONE
        assert completed.completed_at is not None

    def test_complete_nonexistent_raises(self, cx):
        with pytest.raises(ValueError, match="not found"):
            cx.tasks.complete("nonexistent-id")


class TestFollowUps:
    def test_add_follow_up(self, cx):
        task = cx.tasks.create("Main task")
        fu = cx.tasks.add_follow_up(
            task.id,
            action="Review progress",
            due="2025-05-01T00:00:00+00:00",
        )
        assert fu.action == "Review progress"
        assert fu.done is False

        # Verify it's attached to the task
        updated_task = cx.tasks.get(task.id)
        assert len(updated_task.follow_ups) == 1

    def test_multiple_follow_ups(self, cx):
        task = cx.tasks.create("Multi-followup task")
        cx.tasks.add_follow_up(task.id, "Step 1")
        cx.tasks.add_follow_up(task.id, "Step 2")
        cx.tasks.add_follow_up(task.id, "Step 3")

        updated = cx.tasks.get(task.id)
        assert len(updated.follow_ups) == 3

    def test_pending_follow_ups(self, cx):
        task = cx.tasks.create("Task with due follow-up")
        cx.tasks.add_follow_up(
            task.id,
            "Overdue action",
            due="2020-01-01T00:00:00+00:00",  # In the past
        )

        pending = cx.tasks.pending_follow_ups()
        assert len(pending) >= 1
        assert pending[0]["action"] == "Overdue action"


class TestTaskPersistence:
    def test_save_and_reload(self, workspace):
        cx1 = cortexos.init(workspace=workspace)
        cx1.tasks.create("Persistent task", priority=1)
        cx1.save()

        cx2 = cortexos.init(workspace=workspace)
        tasks = cx2.tasks.list()
        assert len(tasks) == 1
        assert tasks[0].summary == "Persistent task"
        assert tasks[0].priority == 1
