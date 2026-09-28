"""Tests for the network-facing tools, driven through a mock transport."""

from __future__ import annotations

import json

import httpx
import pytest
from app.core.config import Settings
from app.tools.base import ToolContext, ToolRequest
from app.tools.github import GitHubClient, GitHubCreateIssueTool, GitHubRepositoryTool
from app.tools.web_search import WebSearchTool

SEARCH_URL = "https://search.example.test/v1/search"


def context_for(**settings_overrides: object) -> ToolContext:
    """Build a context from settings overrides."""
    settings = Settings(_env_file=None, **settings_overrides)  # type: ignore[arg-type]
    return ToolContext(settings=settings)


def transport(handler: object) -> httpx.MockTransport:
    return httpx.MockTransport(handler)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Web search
# --------------------------------------------------------------------------- #


def search_transport(results: list[dict[str, object]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": results}, request=request)

    return httpx.MockTransport(handler)


async def test_search_returns_results() -> None:
    client = httpx.AsyncClient(
        transport=search_transport(
            [{"title": "Postgres", "url": "https://93.184.216.34/pg", "snippet": "docs"}]
        )
    )
    tool = WebSearchTool(client)

    result = await tool.execute(
        ToolRequest(tool="web_search", arguments={"query": "postgres"}),
        context_for(search_api_url=SEARCH_URL),
    )

    assert result.ok is True
    assert result.output["results"][0]["title"] == "Postgres"
    await client.aclose()


async def test_search_marks_every_result_as_untrusted() -> None:
    """Retrieved content is data, and that has to travel with the content."""
    client = httpx.AsyncClient(
        transport=search_transport([{"title": "t", "url": "https://93.184.216.34/x"}])
    )
    tool = WebSearchTool(client)

    result = await tool.execute(
        ToolRequest(tool="web_search", arguments={"query": "q"}),
        context_for(search_api_url=SEARCH_URL),
    )

    assert result.output["results"][0]["untrusted"] is True
    assert "untrusted" in result.output["note"].lower()
    await client.aclose()


async def test_search_drops_results_pointing_at_internal_addresses() -> None:
    """A model must not be handed a URL into the private network."""
    client = httpx.AsyncClient(
        transport=search_transport(
            [
                {"title": "internal", "url": "http://169.254.169.254/latest/meta-data/"},
                {"title": "loopback", "url": "http://127.0.0.1:8000/admin"},
                {"title": "fine", "url": "https://93.184.216.34/ok"},
            ]
        )
    )
    tool = WebSearchTool(client)

    result = await tool.execute(
        ToolRequest(tool="web_search", arguments={"query": "q"}),
        context_for(search_api_url=SEARCH_URL),
    )

    assert result.ok is True
    assert len(result.output["results"]) == 1
    assert result.output["results"][0]["title"] == "fine"
    assert result.output["dropped_unsafe_urls"] == 2
    await client.aclose()


async def test_search_sends_the_query_and_a_bearer_token() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"results": []}, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tool = WebSearchTool(client)

    await tool.execute(
        ToolRequest(tool="web_search", arguments={"query": "hello", "max_results": 3}),
        context_for(search_api_url=SEARCH_URL, search_api_key="search-secret"),
    )

    assert captured["auth"] == "Bearer search-secret"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["query"] == "hello"
    assert body["count"] == 3
    await client.aclose()


async def test_search_without_a_provider_fails_clearly() -> None:
    tool = WebSearchTool()

    result = await tool.execute(
        ToolRequest(tool="web_search", arguments={"query": "q"}), context_for()
    )

    assert result.ok is False
    assert "no search provider" in (result.error or "")


async def test_a_provider_error_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tool = WebSearchTool(client)

    result = await tool.execute(
        ToolRequest(tool="web_search", arguments={"query": "q"}),
        context_for(search_api_url=SEARCH_URL),
    )

    assert result.ok is False
    assert "503" in (result.error or "")
    await client.aclose()


# --------------------------------------------------------------------------- #
# GitHub
# --------------------------------------------------------------------------- #


def github_client(handler: object, *, token: str = "gh-secret-token") -> GitHubClient:
    return GitHubClient(
        token=token,
        base_url="https://api.github.test",
        client=httpx.AsyncClient(transport=transport(handler)),
        secrets=(token,),
    )


async def test_reading_repository_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "full_name": "owner/repo",
                "description": "d",
                "default_branch": "main",
                "stargazers_count": 7,
                "open_issues_count": 2,
                "private": False,
                "html_url": "https://github.com/owner/repo",
            },
            request=request,
        )

    tool = GitHubRepositoryTool(github_client(handler))

    result = await tool.execute(
        ToolRequest(tool="github_repository", arguments={"owner": "owner", "repo": "repo"}),
        context_for(),
    )

    assert result.ok is True
    assert result.output["full_name"] == "owner/repo"
    assert result.output["stars"] == 7


async def test_the_token_never_appears_in_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"full_name": "o/r"}, request=request)

    tool = GitHubRepositoryTool(github_client(handler))

    result = await tool.execute(
        ToolRequest(tool="github_repository", arguments={"owner": "o", "repo": "r"}), context_for()
    )

    assert "gh-secret-token" not in json.dumps(result.model_dump(mode="json"))


async def test_the_token_never_appears_in_an_error_message() -> None:
    """Exception text is a realistic leak path for a credential."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("failed with token gh-secret-token", request=request)

    tool = GitHubRepositoryTool(github_client(handler))

    result = await tool.execute(
        ToolRequest(tool="github_repository", arguments={"owner": "o", "repo": "r"}), context_for()
    )

    assert result.ok is False
    assert "gh-secret-token" not in (result.error or "")


async def test_an_http_error_does_not_echo_the_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403, json={"message": "token gh-secret-token is invalid"}, request=request
        )

    tool = GitHubRepositoryTool(github_client(handler))

    result = await tool.execute(
        ToolRequest(tool="github_repository", arguments={"owner": "o", "repo": "r"}), context_for()
    )

    assert result.ok is False
    assert "gh-secret-token" not in (result.error or "")
    assert "403" in (result.error or "")


async def test_creating_an_issue_is_disabled_by_default() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request should be made when writes are disabled")

    tool = GitHubCreateIssueTool(github_client(handler))

    result = await tool.execute(
        ToolRequest(
            tool="github_create_issue",
            arguments={"owner": "o", "repo": "r", "title": "t"},
        ),
        context_for(),
    )

    assert result.ok is False
    assert "writes are disabled" in (result.error or "")


async def test_creating_an_issue_runs_when_enabled() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        return httpx.Response(
            201,
            json={"number": 12, "title": "t", "html_url": "https://github.com/o/r/issues/12"},
            request=request,
        )

    tool = GitHubCreateIssueTool(github_client(handler))

    result = await tool.execute(
        ToolRequest(
            tool="github_create_issue",
            arguments={"owner": "o", "repo": "r", "title": "t"},
        ),
        context_for(github_tool_allow_writes=True),
    )

    assert result.ok is True
    assert result.output["number"] == 12


def test_github_writes_are_medium_risk_so_they_are_audited() -> None:
    from app.models.tool import RiskLevel

    assert GitHubCreateIssueTool.effective_risk() is RiskLevel.MEDIUM


def test_github_reads_are_low_risk() -> None:
    from app.models.tool import RiskLevel

    assert GitHubRepositoryTool.effective_risk() is RiskLevel.LOW


@pytest.mark.parametrize("bad", ["", "   "])
async def test_an_empty_query_is_rejected_before_any_request(bad: str) -> None:
    tool = WebSearchTool()

    result = await tool.execute(
        ToolRequest(tool="web_search", arguments={"query": bad}),
        context_for(search_api_url=SEARCH_URL),
    )

    assert result.ok is False
