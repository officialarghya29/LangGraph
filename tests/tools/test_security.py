"""Tests for the shared security primitives.

SSRF is the highest-value bug class in any agent that fetches URLs, so these
cases are deliberately adversarial.
"""

from __future__ import annotations

import pytest

from app.core.exceptions import PermissionDeniedError
from app.core.security import redact, validate_outbound_url

# --------------------------------------------------------------------------- #
# Scheme
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "gopher://example.com/",
        "ftp://example.com/x",
        "data:text/plain;base64,AAAA",
        "javascript:alert(1)",
    ],
)
def test_non_http_schemes_are_refused(url: str) -> None:
    with pytest.raises(PermissionDeniedError, match="http and https"):
        validate_outbound_url(url)


def test_a_url_without_a_scheme_is_refused() -> None:
    with pytest.raises(PermissionDeniedError):
        validate_outbound_url("example.com/path")


def test_an_empty_url_is_refused() -> None:
    with pytest.raises(PermissionDeniedError, match="required"):
        validate_outbound_url("   ")


# --------------------------------------------------------------------------- #
# Hosts and addresses
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/admin",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://metadata/computeMetadata/v1/",
    ],
)
def test_blocked_hostnames_are_refused(url: str) -> None:
    with pytest.raises(PermissionDeniedError, match="not reachable"):
        validate_outbound_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000/",
        "http://10.0.0.5/internal",
        "http://192.168.1.1/router",
        "http://172.16.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://0.0.0.0/",
        "http://[::1]/",
        "http://[fe80::1]/",
    ],
)
def test_private_and_reserved_addresses_are_refused(url: str) -> None:
    """The cloud metadata address is the single most valuable SSRF target."""
    with pytest.raises(PermissionDeniedError, match="private or reserved"):
        validate_outbound_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://93.184.216.34/",
        "https://1.1.1.1/",
        "https://8.8.8.8/resolve",
    ],
)
def test_public_addresses_are_allowed(url: str) -> None:
    assert validate_outbound_url(url) == url


def test_an_unresolvable_host_is_refused_rather_than_allowed() -> None:
    """Failing closed: an unresolvable name is not a reason to skip the check."""
    with pytest.raises(PermissionDeniedError):
        validate_outbound_url("http://this-host-does-not-exist.invalid/")


# --------------------------------------------------------------------------- #
# Ports
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("port", [22, 5432, 6379, 9200, 27017])
def test_dangerous_ports_are_refused(port: int) -> None:
    with pytest.raises(PermissionDeniedError, match="port"):
        validate_outbound_url(f"http://93.184.216.34:{port}/")


def test_an_ordinary_port_is_allowed() -> None:
    assert validate_outbound_url("http://93.184.216.34:8080/") == "http://93.184.216.34:8080/"


# --------------------------------------------------------------------------- #
# Opt-out
# --------------------------------------------------------------------------- #


def test_private_destinations_can_be_permitted_explicitly() -> None:
    """Behind a flag, for deployments that legitimately call internal services."""
    assert validate_outbound_url("http://127.0.0.1:9000/", allow_private=True)


def test_the_opt_out_does_not_permit_a_file_url() -> None:
    """Scheme checking is independent of the address policy."""
    with pytest.raises(PermissionDeniedError, match="http and https"):
        validate_outbound_url("file:///etc/passwd", allow_private=True)


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #


def test_redact_removes_every_configured_secret() -> None:
    text = "auth failed: token=abc123 and key=xyz789"

    assert redact(text, ("abc123", "xyz789")) == "auth failed: token=[REDACTED] and key=[REDACTED]"


def test_redact_leaves_unrelated_text_alone() -> None:
    assert redact("nothing sensitive", ("abc123",)) == "nothing sensitive"


def test_redact_ignores_empty_secrets() -> None:
    """An unset secret must not turn every character into a redaction."""
    assert redact("hello", ("",)) == "hello"
