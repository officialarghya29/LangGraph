"""GitHub tools.

Read operations are always available. Writes are off unless explicitly enabled
in configuration, and the write tool is ``MEDIUM`` risk so it is audited without
requiring a human on every call.

The token is used as a request header and appears in no output, no error
message, and no response model. Error details are limited to a status code and
the request path, so a failure cannot echo a credential back to the model.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.core.exceptions import ToolError
from app.core.security import redact
from app.models.tool import AccessMode, RiskLevel
from app.tools.base import Tool, ToolContext

__all__ = [
    "GitHubClient",
    "GitHubCreateIssueTool",
    "GitHubIssueInput",
    "GitHubIssueResult",
    "GitHubRepositoryInput",
    "GitHubRepositoryResult",
    "GitHubRepositoryTool",
]

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 20.0


class GitHubClient:
    """A thin authenticated GitHub API client."""

    def __init__(
        self,
        *,
        token: str,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
        secrets: tuple[str, ...] = (),
    ) -> None:
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = client
        self._secrets = secrets

    async def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        """Perform an API request.

        Raises:
            ToolError: On transport failure or any non-2xx response. The token is
                scrubbed from any message before it is raised.
        """
        url = f"{self._base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        client = self._client or httpx.AsyncClient(timeout=self._timeout)
        try:
            response = await client.request(method, url, headers=headers, json=json_body)
        except httpx.TimeoutException as exc:
            raise ToolError("github request timed out", detail=path) from exc
        except httpx.HTTPError as exc:
            detail = redact(str(exc), self._secrets)
            raise ToolError("github request failed", detail=detail) from exc
        finally:
            if self._client is None:
                await client.aclose()

        if response.status_code >= 400:
            # Status and path only: never the response body, which can echo
            # request headers in some upstream error formats.
            raise ToolError(
                "github returned an error",
                detail=f"HTTP {response.status_code} on {path}",
            )

        try:
            return response.json()
        except ValueError as exc:
            raise ToolError("github returned an unreadable response", detail=path) from exc


# --------------------------------------------------------------------------- #
# Read
# --------------------------------------------------------------------------- #


class GitHubRepositoryInput(BaseModel):
    """Repository to look up."""

    owner: str = Field(min_length=1, max_length=100)
    repo: str = Field(min_length=1, max_length=100)


class GitHubRepositoryResult(BaseModel):
    """Selected repository metadata."""

    full_name: str
    description: str = ""
    default_branch: str = ""
    stars: int = 0
    open_issues: int = 0
    private: bool = False
    html_url: str = ""


class GitHubRepositoryTool(Tool[GitHubRepositoryInput, GitHubRepositoryResult]):
    """Reads repository metadata."""

    name = "github_repository"
    description = "Read public or authorised repository metadata from GitHub"
    access_mode = AccessMode.READ
    risk_level = RiskLevel.LOW
    input_model = GitHubRepositoryInput
    output_model = GitHubRepositoryResult

    def __init__(self, client: GitHubClient) -> None:
        self._client = client

    async def run(
        self, payload: GitHubRepositoryInput, context: ToolContext
    ) -> GitHubRepositoryResult:
        """Fetch repository metadata."""
        del context
        data = await self._client.request("GET", f"/repos/{payload.owner}/{payload.repo}")

        if not isinstance(data, dict):
            raise ToolError("github returned an unexpected payload")

        return GitHubRepositoryResult(
            full_name=data.get("full_name", f"{payload.owner}/{payload.repo}"),
            description=data.get("description") or "",
            default_branch=data.get("default_branch") or "",
            stars=int(data.get("stargazers_count") or 0),
            open_issues=int(data.get("open_issues_count") or 0),
            private=bool(data.get("private")),
            html_url=data.get("html_url") or "",
        )


# --------------------------------------------------------------------------- #
# Write
# --------------------------------------------------------------------------- #


class GitHubIssueInput(BaseModel):
    """An issue to open."""

    owner: str = Field(min_length=1, max_length=100)
    repo: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=256)
    body: str = Field(default="", max_length=20_000)


class GitHubIssueResult(BaseModel):
    """The issue that was created."""

    number: int
    title: str
    html_url: str


class GitHubCreateIssueTool(Tool[GitHubIssueInput, GitHubIssueResult]):
    """Opens an issue. Disabled unless writes are explicitly enabled."""

    name = "github_create_issue"
    description = "Open a GitHub issue. Disabled unless GITHUB_TOOL_ALLOW_WRITES is set"
    access_mode = AccessMode.WRITE
    input_model = GitHubIssueInput
    output_model = GitHubIssueResult

    def __init__(self, client: GitHubClient) -> None:
        self._client = client

    async def run(self, payload: GitHubIssueInput, context: ToolContext) -> GitHubIssueResult:
        """Create the issue.

        Raises:
            ToolError: If writes are disabled.
        """
        if not context.settings.github_tool_allow_writes:
            raise ToolError(
                "github writes are disabled",
                detail="set GITHUB_TOOL_ALLOW_WRITES to enable them",
            )

        logger.info(
            "github.create_issue",
            extra={"repo": f"{payload.owner}/{payload.repo}", "task_id": context.task_id},
        )

        data = await self._client.request(
            "POST",
            f"/repos/{payload.owner}/{payload.repo}/issues",
            json_body={"title": payload.title, "body": payload.body},
        )

        if not isinstance(data, dict):
            raise ToolError("github returned an unexpected payload")

        return GitHubIssueResult(
            number=int(data.get("number") or 0),
            title=data.get("title") or payload.title,
            html_url=data.get("html_url") or "",
        )
