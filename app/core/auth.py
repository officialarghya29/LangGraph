"""Identity and authorization.

Identity resolution is one function, so there is exactly one place that decides
who the caller is. Everything downstream — ownership checks, tool permissions,
rate-limit buckets, memory scoping — takes a :class:`Principal` and never
re-derives identity from a header, a body field, or a path parameter.

Three modes, and the difference between them matters:

- **Authentication enabled** — a bearer JWT is required and verified. An absent,
  malformed, expired, or wrongly-signed token is a 401. Nothing else is accepted.
- **Authentication disabled, header trusted** — the ``X-User-Id`` header is taken
  as the principal. This exists so the system can be exercised locally without an
  identity provider. It is self-asserted, so it is refused outright when
  ``APP_ENV`` is ``production``.
- **Authentication disabled, header not trusted** — every caller is one anonymous
  principal. Least privilege by default: if identity is not established, the
  caller does not get to claim one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Annotated, Any

import jwt
from fastapi import Depends, Header, Request
from jwt import InvalidTokenError

from app.core.config import Settings, get_settings
from app.core.exceptions import AuthenticationError, PermissionDeniedError

__all__ = [
    "ANONYMOUS_USER_ID",
    "AUTHORIZATION_HEADER",
    "IDENTITY_HEADER",
    "Principal",
    "PrincipalDep",
    "SettingsDep",
    "decode_token",
    "ensure_owner",
    "get_settings_dep",
    "issue_token",
    "require_scopes",
    "resolve_principal",
]

logger = logging.getLogger(__name__)

#: Header carrying a caller's own identity while authentication is disabled.
IDENTITY_HEADER = "X-User-Id"
#: Standard bearer-token header.
AUTHORIZATION_HEADER = "Authorization"
#: The single principal used when nobody has identified themselves.
ANONYMOUS_USER_ID = "anonymous"

#: Bearer prefix, compared case-insensitively as the specification requires.
_BEARER_PREFIX = "bearer "


@dataclass(frozen=True, slots=True)
class Principal:
    """An authenticated caller.

    Deliberately small. It carries an identity and a set of granted scopes, and
    nothing that a downstream component could use to reach around the
    authorization checks built on top of it.
    """

    user_id: str
    #: Whether the identity was cryptographically established. False for the
    #: anonymous fallback and for a trusted development header.
    authenticated: bool = False
    scopes: frozenset[str] = field(default_factory=frozenset)
    #: Free-form claims a downstream policy may consult (tenant, plan, ...).
    claims: dict[str, Any] = field(default_factory=dict)

    @property
    def is_anonymous(self) -> bool:
        """Return whether this is the unauthenticated fallback principal."""
        return self.user_id == ANONYMOUS_USER_ID

    def has_scope(self, scope: str) -> bool:
        """Return whether this principal holds a scope.

        ``*`` is honoured as a wildcard, which is how a service account is
        granted broad access without enumerating every scope.
        """
        return "*" in self.scopes or scope in self.scopes


def get_settings_dep() -> Settings:
    """Return process settings.

    A named function rather than passing :func:`app.core.config.get_settings`
    directly, so a test can override it with ``dependency_overrides``.
    """
    return get_settings()


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]


def decode_token(token: str, settings: Settings) -> dict[str, Any]:
    """Verify and decode a JWT.

    Args:
        token: The compact-serialised token.
        settings: Settings holding the signing secret and expected claims.

    Returns:
        The verified claims.

    Raises:
        AuthenticationError: If the token is invalid for any reason. Every
            failure — bad signature, expired, wrong audience, wrong algorithm —
            returns the same message, because distinguishing them tells an
            attacker which part of a forgery to fix.
    """
    if settings.jwt_secret is None:
        raise AuthenticationError("authentication is enabled but no signing key is configured")

    try:
        return jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            # Pinned exactly. Without an explicit list, a token with "alg": "none"
            # or a symmetric algorithm substituted for an asymmetric one would be
            # accepted, which is the classic JWT forgery.
            algorithms=[settings.jwt_algorithm],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            options={
                "require": ["exp", "sub"],
                "verify_exp": True,
                "verify_aud": settings.jwt_audience is not None,
                "verify_iss": settings.jwt_issuer is not None,
            },
        )
    except InvalidTokenError as exc:
        logger.warning("auth.token_rejected", extra={"reason": type(exc).__name__})
        raise AuthenticationError("the bearer token is not valid") from exc


def issue_token(
    user_id: str,
    settings: Settings,
    *,
    scopes: frozenset[str] | None = None,
    expires_in_seconds: int = 3600,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Mint a signed token for ``user_id``.

    Used by tests and by the local development helper. A real deployment expects
    tokens from an identity provider, not from this function.

    Args:
        user_id: The subject.
        settings: Settings holding the signing secret.
        scopes: Scopes to grant. Serialised as a space-separated string, which is
            the conventional form and survives providers that only emit strings.
        expires_in_seconds: Token lifetime.
        extra_claims: Additional claims to include.

    Returns:
        A compact-serialised JWT.

    Raises:
        AuthenticationError: If no signing key is configured.
    """
    import time

    if settings.jwt_secret is None:
        raise AuthenticationError("no signing key is configured")

    now = int(time.time())
    claims: dict[str, Any] = {
        "sub": user_id,
        "iat": now,
        "exp": now + expires_in_seconds,
        "scope": " ".join(sorted(scopes or frozenset())),
    }
    if settings.jwt_issuer is not None:
        claims["iss"] = settings.jwt_issuer
    if settings.jwt_audience is not None:
        claims["aud"] = settings.jwt_audience
    claims.update(extra_claims or {})

    return jwt.encode(
        claims, settings.jwt_secret.get_secret_value(), algorithm=settings.jwt_algorithm
    )


def _principal_from_claims(claims: dict[str, Any]) -> Principal:
    """Build a principal from verified claims."""
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        # A verified token with no usable subject is unusable: every ownership
        # check is keyed on it.
        raise AuthenticationError("the bearer token carries no subject")

    raw_scope = claims.get("scope") or claims.get("scp") or ""
    if isinstance(raw_scope, str):
        scopes = frozenset(part for part in raw_scope.split() if part)
    elif isinstance(raw_scope, list):
        scopes = frozenset(str(part) for part in raw_scope)
    else:
        scopes = frozenset()

    return Principal(
        user_id=subject,
        authenticated=True,
        scopes=scopes,
        claims={key: value for key, value in claims.items() if key not in {"sub", "scope"}},
    )


async def resolve_principal(
    settings: SettingsDep,
    authorization: Annotated[str | None, Header()] = None,
    x_user_id: Annotated[str | None, Header()] = None,
) -> Principal:
    """Resolve the caller's identity.

    Args:
        settings: Application settings.
        authorization: The ``Authorization`` header, if present.
        x_user_id: The development identity header, if present.

    Returns:
        The resolved principal.

    Raises:
        AuthenticationError: If authentication is enabled and the bearer token is
            absent or invalid. A header alone is never sufficient in that mode.
    """
    if settings.auth_enabled:
        if not authorization or not authorization.lower().startswith(_BEARER_PREFIX):
            raise AuthenticationError("a bearer token is required")
        token = authorization[len(_BEARER_PREFIX) :].strip()
        if not token:
            raise AuthenticationError("a bearer token is required")
        return _principal_from_claims(decode_token(token, settings))

    if settings.trust_identity_header and x_user_id:
        # Self-asserted. Acceptable only because the production validator refuses
        # to let this setting be true outside development.
        return Principal(user_id=x_user_id.strip(), authenticated=False)

    return Principal(user_id=ANONYMOUS_USER_ID, authenticated=False)


PrincipalDep = Annotated[Principal, Depends(resolve_principal)]


def ensure_owner(principal: Principal, owner_id: str) -> None:
    """Raise unless ``principal`` owns the resource.

    Raises:
        PermissionDeniedError: If the principal does not own the resource.
    """
    if principal.user_id != owner_id:
        raise PermissionDeniedError("this resource belongs to another principal")


def require_scopes(*required: str) -> Any:
    """Build a dependency that requires the caller to hold every given scope.

    Args:
        *required: Scopes the caller must hold.

    Returns:
        A dependency that yields the principal, or raises.
    """

    async def dependency(principal: PrincipalDep) -> Principal:
        missing = [scope for scope in required if not principal.has_scope(scope)]
        if missing:
            raise PermissionDeniedError(
                "the caller lacks a required scope", detail=", ".join(sorted(missing))
            )
        return principal

    return dependency


async def current_principal(request: Request) -> Principal:
    """Return the principal already resolved for this request.

    Lets middleware and stream handlers — which do not participate in FastAPI's
    dependency injection — reuse the same resolution instead of re-deriving
    identity. Falls back to resolving from the request when the dependency has
    not run, which is the case for a raw ASGI scope.

    Args:
        request: The incoming request.

    Returns:
        The principal for this request.
    """
    cached = getattr(request.state, "principal", None)
    if isinstance(cached, Principal):
        return cached

    settings = get_settings()
    principal = await resolve_principal(
        settings,
        request.headers.get(AUTHORIZATION_HEADER),
        request.headers.get(IDENTITY_HEADER),
    )
    request.state.principal = principal
    return principal
