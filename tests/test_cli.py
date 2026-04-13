"""Tests for CLI commands."""

import shutil
import tempfile

import pytest
from click.testing import CliRunner

from cortexos.cli.main import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def workspace():
    tmpdir = tempfile.mkdtemp(prefix="cortexos_test_cli_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestCLIStore:
    def test_store_basic(self, runner, workspace):
        result = runner.invoke(cli, ["-w", workspace, "store", "Test memory"])
        assert result.exit_code == 0
        assert "Stored:" in result.output

    def test_store_with_type(self, runner, workspace):
        result = runner.invoke(
            cli, ["-w", workspace, "store", "Decision made", "-t", "decision"]
        )
        assert result.exit_code == 0
        assert "decision" in result.output


class TestCLIRecall:
    def test_recall_empty(self, runner, workspace):
        result = runner.invoke(cli, ["-w", workspace, "recall", "anything"])
        assert result.exit_code == 0
        assert "No relevant memories" in result.output

    def test_recall_with_data(self, runner, workspace):
        runner.invoke(cli, ["-w", workspace, "store", "Python is great"])
        result = runner.invoke(cli, ["-w", workspace, "recall", "Python"])
        assert result.exit_code == 0


class TestCLIZones:
    def test_zones_list_empty(self, runner, workspace):
        result = runner.invoke(cli, ["-w", workspace, "zones", "list"])
        assert result.exit_code == 0

    def test_zones_create(self, runner, workspace):
        result = runner.invoke(
            cli, ["-w", workspace, "zones", "create", "test-zone", "-s", "For testing"]
        )
        assert result.exit_code == 0
        assert "Created zone: test-zone" in result.output

    def test_zones_stats(self, runner, workspace):
        runner.invoke(cli, ["-w", workspace, "zones", "create", "sz"])
        result = runner.invoke(cli, ["-w", workspace, "zones", "stats", "sz"])
        assert result.exit_code == 0
        assert "name: sz" in result.output


class TestCLITasks:
    def test_task_create(self, runner, workspace):
        result = runner.invoke(
            cli, ["-w", workspace, "task", "create", "New task"]
        )
        assert result.exit_code == 0
        assert "Created task:" in result.output

    def test_task_list_empty(self, runner, workspace):
        result = runner.invoke(cli, ["-w", workspace, "task", "list"])
        assert result.exit_code == 0
        assert "No tasks" in result.output

    def test_task_list_with_data(self, runner, workspace):
        runner.invoke(cli, ["-w", workspace, "task", "create", "Task A"])
        result = runner.invoke(cli, ["-w", workspace, "task", "list"])
        assert result.exit_code == 0
        assert "Task A" in result.output


class TestCLIStats:
    def test_stats(self, runner, workspace):
        result = runner.invoke(cli, ["-w", workspace, "stats"])
        assert result.exit_code == 0
        assert "total_entries" in result.output


class TestCLILifecycle:
    def test_consolidate(self, runner, workspace):
        result = runner.invoke(cli, ["-w", workspace, "consolidate"])
        assert result.exit_code == 0
        assert "Consolidation complete" in result.output

    def test_garden(self, runner, workspace):
        result = runner.invoke(cli, ["-w", workspace, "garden"])
        assert result.exit_code == 0
        assert "Garden complete" in result.output
