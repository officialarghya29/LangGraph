"""Security primitives shared across the application.

The main job here is SSRF defence. Any tool that fetches a URL supplied by, or
influenced by, a language model is a server-side request forgery primitive: the
model can be talked into requesting ``http://169.254.169.254/`` and the server
will happily fetch it. URLs are therefore treated as untrusted input and
resolved before they are used.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

from app.core.exceptions import PermissionDeniedError

__all__ = ["ALLOWED_SCHEMES", "redact", "validate_outbound_url"]

#: Schemes that may be fetched. ``file`` and ``gopher`` are excluded by design.
ALLOWED_SCHEMES = frozenset({"http", "https"})

#: Hostnames that are always refused. Cloud metadata endpoints are the highest
#: value SSRF target because they hand out credentials.
BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "metadata",
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
    }
)

#: Ports that should never be reachable from an agent.
BLOCKED_PORTS = frozenset({22, 23, 25, 445, 1433, 3306, 5432, 6379, 9200, 11211, 27017})


def _is_blocked_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return whether an address is one an agent must not reach."""
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def validate_outbound_url(url: str, *, allow_private: bool = False) -> str:
    """Validate a URL before any outbound request is made.

    Checks the scheme, the hostname, the port, and — critically — the resolved
    IP addresses. A hostname that resolves to a private address is refused even
    if the name itself looks innocent, which is what defeats DNS-rebinding
    style bypasses.

    Args:
        url: The URL to validate.
        allow_private: Permit private and loopback destinations. Defaults to
            refusing them.

    Returns:
        The validated URL.

    Raises:
        PermissionDeniedError: If the URL may not be fetched.
    """
    if not url or not url.strip():
        raise PermissionDeniedError("a URL is required")

    parsed = urlparse(url.strip())

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise PermissionDeniedError(
            "only http and https URLs may be fetched", detail=f"scheme={parsed.scheme or '(none)'}"
        )

    hostname = parsed.hostname
    if not hostname:
        raise PermissionDeniedError("the URL has no host")

    if hostname.lower() in BLOCKED_HOSTNAMES:
        raise PermissionDeniedError("that host is not reachable", detail=hostname)

    if parsed.port is not None and parsed.port in BLOCKED_PORTS:
        raise PermissionDeniedError("that port is not reachable", detail=str(parsed.port))

    if allow_private:
        return url

    for address in _resolve(hostname):
        if _is_blocked_address(address):
            raise PermissionDeniedError(
                "that host resolves to a private or reserved address", detail=hostname
            )

    return url


def _resolve(hostname: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Resolve a hostname to IP addresses.

    A literal IP is parsed directly; anything else is resolved through DNS.

    Raises:
        PermissionDeniedError: If the host cannot be resolved. Failing closed is
            deliberate: an unresolvable host is not a reason to skip the check.
    """
    try:
        return [ipaddress.ip_address(hostname)]
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise PermissionDeniedError("that host could not be resolved", detail=hostname) from exc

    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        try:
            addresses.append(ipaddress.ip_address(info[4][0]))
        except ValueError:  # pragma: no cover - defensive
            continue

    if not addresses:
        raise PermissionDeniedError("that host could not be resolved", detail=hostname)
    return addresses


def redact(text: str, secrets: tuple[str, ...]) -> str:
    """Replace every configured secret in ``text`` with a placeholder.

    Applied to error messages and log lines, because a credential appearing in
    an exception string is one of the most common ways a secret leaks.

    Args:
        text: The text to sanitise.
        secrets: Secret values to remove.

    Returns:
        The sanitised text.
    """
    for secret in secrets:
        if secret and secret in text:
            text = text.replace(secret, "[REDACTED]")
    return text
