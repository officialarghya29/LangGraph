"""Tests for the confined filesystem tools."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.exceptions import ToolError
from app.tools.base import ToolContext, ToolRequest
from app.tools.filesystem import (
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
    resolve_within_roots,
)


def settings_for(root: str, *, max_bytes: int = 1024) -> Settings:
    return Settings(
        _env_file=None,
        filesystem_allowed_roots=root,
        filesystem_max_file_bytes=max_bytes,
    )


def context_for(root: str, *, max_bytes: int = 1024) -> ToolContext:
    return ToolContext(settings=settings_for(root, max_bytes=max_bytes))


# --------------------------------------------------------------------------- #
# Containment
# --------------------------------------------------------------------------- #


def test_a_relative_path_resolves_inside_the_root(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path)) / "workspace"
    root.mkdir()

    resolved = resolve_within_roots("notes.txt", (root,))

    assert resolved == (root / "notes.txt")


def test_traversal_out_of_the_root_is_refused(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path)) / "workspace"
    root.mkdir()

    with pytest.raises(ToolError, match="outside the allowed roots"):
        resolve_within_roots("../secrets.txt", (root,))


def test_deep_traversal_is_refused(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path)) / "workspace"
    root.mkdir()

    with pytest.raises(ToolError, match="outside the allowed roots"):
        resolve_within_roots("a/b/../../../etc/passwd", (root,))


def test_an_absolute_path_outside_the_root_is_refused(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path)) / "workspace"
    root.mkdir()

    with pytest.raises(ToolError, match="outside the allowed roots"):
        resolve_within_roots("/etc/passwd", (root,))


def test_a_symlink_pointing_outside_the_root_is_refused(tmp_path: object) -> None:
    """Resolving before checking is what defeats a symlink escape."""
    from pathlib import Path

    base = Path(str(tmp_path))
    root = base / "workspace"
    root.mkdir()
    outside = base / "outside.txt"
    outside.write_text("secret")
    (root / "escape.txt").symlink_to(outside)

    with pytest.raises(ToolError, match="outside the allowed roots"):
        resolve_within_roots("escape.txt", (root,))


def test_a_symlink_pointing_inside_the_root_is_allowed(tmp_path: object) -> None:
    from pathlib import Path

    base = Path(str(tmp_path))
    root = base / "workspace"
    root.mkdir()
    (root / "real.txt").write_text("fine")
    (root / "link.txt").symlink_to(root / "real.txt")

    assert resolve_within_roots("link.txt", (root,)) == root / "real.txt"


def test_no_configured_root_is_refused(tmp_path: object) -> None:
    with pytest.raises(ToolError, match="no filesystem roots"):
        resolve_within_roots("anything.txt", ())


def test_a_null_byte_is_refused(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))

    with pytest.raises(ToolError, match="null byte"):
        resolve_within_roots("bad\x00name.txt", (root,))


def test_an_over_long_path_is_refused(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))

    with pytest.raises(ToolError, match="too long"):
        resolve_within_roots("a" * 5000, (root,))


# --------------------------------------------------------------------------- #
# Read
# --------------------------------------------------------------------------- #


async def test_reading_a_file_inside_the_root(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))
    (root / "hello.txt").write_text("hello world")

    result = await ReadFileTool().execute(
        ToolRequest(tool="read_file", arguments={"path": "hello.txt"}), context_for(str(root))
    )

    assert result.ok is True
    assert result.output["content"] == "hello world"


async def test_reading_outside_the_root_fails(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))

    result = await ReadFileTool().execute(
        ToolRequest(tool="read_file", arguments={"path": "../../etc/passwd"}),
        context_for(str(root)),
    )

    assert result.ok is False
    assert "outside the allowed roots" in (result.error or "")


async def test_a_large_file_is_truncated(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))
    (root / "big.txt").write_text("x" * 5000)

    result = await ReadFileTool().execute(
        ToolRequest(tool="read_file", arguments={"path": "big.txt"}),
        context_for(str(root), max_bytes=100),
    )

    assert result.ok is True
    assert result.output["truncated"] is True
    assert result.output["bytes_read"] == 100


async def test_reading_a_missing_file_fails(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))

    result = await ReadFileTool().execute(
        ToolRequest(tool="read_file", arguments={"path": "nope.txt"}), context_for(str(root))
    )

    assert result.ok is False
    assert "no such file" in (result.error or "")


async def test_reading_a_directory_fails(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))

    result = await ReadFileTool().execute(
        ToolRequest(tool="read_file", arguments={"path": "."}), context_for(str(root))
    )

    assert result.ok is False
    assert "directory" in (result.error or "")


# --------------------------------------------------------------------------- #
# Write
# --------------------------------------------------------------------------- #


async def test_writing_a_file_inside_the_root(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))

    result = await WriteFileTool().execute(
        ToolRequest(tool="write_file", arguments={"path": "out.txt", "content": "data"}),
        context_for(str(root)),
    )

    assert result.ok is True
    assert (root / "out.txt").read_text() == "data"


async def test_writing_outside_the_root_fails(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path)) / "workspace"
    root.mkdir()

    result = await WriteFileTool().execute(
        ToolRequest(tool="write_file", arguments={"path": "../escape.txt", "content": "x"}),
        context_for(str(root)),
    )

    assert result.ok is False
    assert not (root.parent / "escape.txt").exists()


async def test_an_oversized_write_is_refused(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))

    result = await WriteFileTool().execute(
        ToolRequest(tool="write_file", arguments={"path": "big.txt", "content": "x" * 500}),
        context_for(str(root), max_bytes=100),
    )

    assert result.ok is False
    assert "maximum file size" in (result.error or "")


def test_writing_is_classified_as_a_write() -> None:
    """A write must not be able to pass as a low-risk read."""
    from app.models.tool import AccessMode, RiskLevel

    assert WriteFileTool.access_mode is AccessMode.WRITE
    assert WriteFileTool.effective_risk() is RiskLevel.MEDIUM


# --------------------------------------------------------------------------- #
# List
# --------------------------------------------------------------------------- #


async def test_listing_a_directory(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))
    (root / "a.txt").write_text("a")
    (root / "sub").mkdir()

    result = await ListDirectoryTool().execute(
        ToolRequest(tool="list_directory", arguments={"path": "."}), context_for(str(root))
    )

    assert result.ok is True
    assert "a.txt" in result.output["entries"]
    assert "sub/" in result.output["entries"]


async def test_listing_outside_the_root_fails(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path)) / "workspace"
    root.mkdir()

    result = await ListDirectoryTool().execute(
        ToolRequest(tool="list_directory", arguments={"path": "/etc"}), context_for(str(root))
    )

    assert result.ok is False
