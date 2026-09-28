"""Filesystem tools.

The tool never sees the machine's filesystem. Every path is resolved and then
checked against an allow-listed root, so a model cannot read ``/etc/shadow`` by
asking for it, and cannot escape with ``../../``.

Symlink escapes are handled by resolving before checking: a symlink that points
outside the root resolves to its target, and the target fails the containment
check. Checking the raw path instead would be trivially bypassable.

The three tools are registered separately (``read_file``, ``write_file``,
``list_directory``) rather than as one switchable tool, so an agent's allow-list
can grant reading without granting writing. The coding agent holds all three; a
read-only role would hold only the first.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.exceptions import ToolError
from app.models.tool import AccessMode, RiskLevel
from app.tools.base import Tool, ToolContext

__all__ = [
    "DirectoryListing",
    "FileContent",
    "ListDirectoryTool",
    "ListInput",
    "ReadFileTool",
    "ReadInput",
    "WriteFileTool",
    "WriteInput",
    "WriteOutcome",
    "resolve_within_roots",
]

logger = logging.getLogger(__name__)

#: Refuse absurd paths before touching the filesystem.
MAX_PATH_LENGTH = 4096


def resolve_within_roots(raw_path: str, roots: tuple[Path, ...]) -> Path:
    """Resolve a user-supplied path and confirm it stays inside an allowed root.

    Args:
        raw_path: The requested path, relative or absolute.
        roots: Allow-listed roots the path must fall within.

    Returns:
        The resolved absolute path.

    Raises:
        ToolError: If the path is malformed, escapes every root, or no root is
            configured.
    """
    if not raw_path or not raw_path.strip():
        raise ToolError("a path is required")
    if len(raw_path) > MAX_PATH_LENGTH:
        raise ToolError("the path is too long")
    if "\x00" in raw_path:
        raise ToolError("the path contains a null byte")

    if not roots:
        raise ToolError("no filesystem roots are configured")

    candidate = Path(raw_path.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = roots[0] / candidate

    # resolve() follows symlinks, which is what makes the containment check
    # meaningful rather than cosmetic.
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as exc:
        raise ToolError("the path could not be resolved", detail=str(exc)) from exc

    for root in roots:
        if resolved == root or root in resolved.parents:
            return resolved

    raise ToolError("the path is outside the allowed roots")


class ReadInput(BaseModel):
    """Request to read one file."""

    path: str = Field(min_length=1)


class FileContent(BaseModel):
    """The contents of a file, possibly truncated."""

    path: str
    content: str
    bytes_read: int
    truncated: bool


class WriteInput(BaseModel):
    """Request to write one file."""

    path: str = Field(min_length=1)
    content: str


class WriteOutcome(BaseModel):
    """The result of a write."""

    path: str
    bytes_written: int


class ListInput(BaseModel):
    """Request to list one directory."""

    path: str = "."


class DirectoryListing(BaseModel):
    """The entries in a directory."""

    path: str
    entries: list[str] = Field(default_factory=list)


class ReadFileTool(Tool[ReadInput, FileContent]):
    """Reads a file from within an allowed root."""

    name = "read_file"
    description = "Read a UTF-8 text file from within the allowed workspace"
    access_mode = AccessMode.READ
    risk_level = RiskLevel.LOW
    input_model = ReadInput
    output_model = FileContent

    async def run(self, payload: ReadInput, context: ToolContext) -> FileContent:
        """Read the requested file, up to the configured size limit."""
        path = resolve_within_roots(payload.path, context.settings.allowed_roots)
        limit = context.settings.filesystem_max_file_bytes

        if not path.exists():
            raise ToolError("no such file", detail=payload.path)
        if path.is_dir():
            raise ToolError("that path is a directory, not a file", detail=payload.path)

        try:
            with path.open("rb") as handle:
                data = handle.read(limit + 1)
        except OSError as exc:
            raise ToolError("the file could not be read", detail=exc.strerror) from exc

        truncated = len(data) > limit
        data = data[:limit]

        logger.info(
            "filesystem.read",
            extra={"path": str(path), "bytes": len(data), "truncated": truncated},
        )

        return FileContent(
            path=str(path),
            content=data.decode("utf-8", errors="replace"),
            bytes_read=len(data),
            truncated=truncated,
        )


class WriteFileTool(Tool[WriteInput, WriteOutcome]):
    """Writes a file inside an allowed root.

    MEDIUM risk: a write is auditable but not destructive, so it does not need
    human approval. Deleting is not offered at all.
    """

    name = "write_file"
    description = "Write a UTF-8 text file inside the allowed workspace"
    access_mode = AccessMode.WRITE
    input_model = WriteInput
    output_model = WriteOutcome

    async def run(self, payload: WriteInput, context: ToolContext) -> WriteOutcome:
        """Write the file, refusing anything over the size limit."""
        path = resolve_within_roots(payload.path, context.settings.allowed_roots)
        encoded = payload.content.encode("utf-8")
        limit = context.settings.filesystem_max_file_bytes

        if len(encoded) > limit:
            raise ToolError(
                "the content exceeds the maximum file size",
                detail=f"{len(encoded)} > {limit} bytes",
            )

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(encoded)
        except OSError as exc:
            raise ToolError("the file could not be written", detail=exc.strerror) from exc

        logger.info("filesystem.write", extra={"path": str(path), "bytes": len(encoded)})

        return WriteOutcome(path=str(path), bytes_written=len(encoded))


class ListDirectoryTool(Tool[ListInput, DirectoryListing]):
    """Lists one directory inside an allowed root."""

    name = "list_directory"
    description = "List the entries of a directory inside the allowed workspace"
    access_mode = AccessMode.READ
    risk_level = RiskLevel.LOW
    input_model = ListInput
    output_model = DirectoryListing

    async def run(self, payload: ListInput, context: ToolContext) -> DirectoryListing:
        """List the directory's immediate children."""
        path = resolve_within_roots(payload.path, context.settings.allowed_roots)

        if not path.is_dir():
            raise ToolError("no such directory", detail=payload.path)

        try:
            entries = sorted(
                entry.name + ("/" if entry.is_dir() else "") for entry in path.iterdir()
            )
        except OSError as exc:
            raise ToolError("the directory could not be listed", detail=exc.strerror) from exc

        logger.info("filesystem.list", extra={"path": str(path), "entries": len(entries)})

        return DirectoryListing(path=str(path), entries=entries)
