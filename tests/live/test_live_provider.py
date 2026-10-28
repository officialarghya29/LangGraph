"""One real call to a real vendor endpoint, when a key is configured.

Every other suite in this project talks to a scripted server. That covers the
request the adapter builds and the response it parses, and it cannot cover the
things only the vendor decides: whether the model name exists today, whether the
payload shape is still accepted for that model, and whether the fields this
project treats as optional are in fact optional. Those are the failure modes that
turn a green test run into a red first request, and no amount of mocking reaches
them.

The trade is that a live call costs money, needs a network, and is not
deterministic — so it does not belong in the default gate. This module skips
itself unless ``LLM_API_KEY`` resolves to a real value, which means:

* with no key, the run is green and says why in one skipped line;
* with a key, the call is made and the result is asserted.

It is deliberately one call with a short, cheap, checkable answer. A broader live
suite would be a second product; a single call answers the question that matters,
which is "is this configuration accepted at all".

Run it with a key, explicitly::

    LLM_API_KEY=... python -m pytest tests/live -m live
"""

from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.services.llm import Message, Role, build_llm_provider

pytestmark = pytest.mark.live

#: The cheapest prompt that still proves a round trip: one short instruction, one
#: word expected back, and a ceiling low enough that a runaway response is capped.
PROMPT = "Reply with exactly the two characters: ok"
ANSWER = "ok"


def live_provider() -> object:
    """Build the provider configured in this environment, or skip.

    Returns:
        A provider built from the settings, ready to call.

    Raises:
        pytest.skip.Exception: If no credential is configured, which is the normal
            case in CI and on a fresh checkout.
    """
    settings = get_settings()
    if settings.llm_api_key is None:
        pytest.skip(
            "no LLM_API_KEY is configured, so no vendor endpoint can be called; "
            "the scripted transport tests cover the request and response contract"
        )
    return build_llm_provider(settings)


async def test_the_configured_model_accepts_the_request_this_adapter_sends() -> None:
    """The one claim the mocked suites cannot make.

    What this establishes, if it passes: the resolved model name exists for this
    account, the endpoint accepts the payload — including whichever sampling
    fields the adapter decided are legal for that model — and the response has the
    shape the adapter parses. What it does not establish is anything about cost,
    latency, or quality, and it deliberately asserts none of those.
    """
    provider = live_provider()  # type: ignore[assignment]
    settings = get_settings()

    response = await provider.ainvoke(
        [Message(role=Role.USER, content=PROMPT)],
        max_tokens=16,
    )

    assert response.content.strip().lower().startswith(ANSWER), (
        f"{settings.llm_provider}/{settings.llm_model_name} answered "
        f"{response.content[:120]!r}, which is not the requested {ANSWER!r}. A model "
        "that ignores this instruction is not necessarily broken, but the adapter is "
        "no longer talking to the model the configuration names."
    )
    assert response.model, "the vendor did not report which model answered"
    assert response.usage.total_tokens > 0, "the vendor reported no token usage"


async def test_the_configured_provider_is_the_one_that_answered() -> None:
    """A silent redirect to another model is worth knowing about.

    Vendors alias and retire names, and an alias resolves to a snapshot whose name
    differs from the one requested. That is fine — but it should be visible, not
    inferred, because it is the difference between "my model choice took effect"
    and "it quietly did not".
    """
    provider = live_provider()  # type: ignore[assignment]
    settings = get_settings()

    response = await provider.ainvoke(
        [Message(role=Role.USER, content=PROMPT)],
        max_tokens=16,
    )

    assert response.model.startswith(settings.llm_model_name.split("-")[0]), (
        f"requested {settings.llm_model_name!r} from {settings.llm_provider}, "
        f"and {response.model!r} answered; those are different model families"
    )
