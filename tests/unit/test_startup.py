"""Startup checks that must not become startup failures.

The catalogue check is the only thing the application does at startup that talks
to a vendor, and it exists to turn "the model was retired" from a 404 at the first
paid call into a log line at boot. These tests pin the part that matters: it warns
when it knows, and says nothing when it does not. A check that guessed would be
worse than no check, because a warning that fires on every deployment is a warning
nobody reads.
"""

from __future__ import annotations

import logging

import pytest

from app.core.config import Settings
from app.main import _warn_if_model_is_missing
from app.services.llm import FakeLLMProvider, LLMError


class Catalogue(FakeLLMProvider):
    """A provider that answers with a fixed catalogue, or refuses to answer."""

    def __init__(self, models: tuple[str, ...] | None) -> None:
        super().__init__()
        self._models = models

    async def alist_models(self) -> tuple[str, ...]:
        """Return the scripted catalogue, or fail the way a real one would."""
        if self._models is None:
            raise LLMError("provider refused the model listing", detail="HTTP 503")
        return self._models


def settings_for(model: str) -> Settings:
    """Return settings naming ``model`` on the OpenAI-compatible surface."""
    return Settings(
        _env_file=None,
        llm_provider="openai",
        # ``llm_model``, not the resolved ``llm_model_name``: the latter is a
        # property, and passing it as a field would be ignored in favour of the
        # provider default — which is how the first version of this test passed a
        # retired name and still saw a live one.
        llm_model=model,
        llm_api_key="test-key",
    )


async def test_a_retired_model_is_warned_about_at_startup(caplog: pytest.LogCaptureFixture) -> None:
    """The warning names the model and the live names closest to it."""
    provider = Catalogue(("gpt-6.1-sol", "gpt-6-luna"))

    with caplog.at_level(logging.WARNING):
        await _warn_if_model_is_missing(provider, settings_for("gpt-6.1-sun"))  # type: ignore[arg-type]

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.getMessage() == "startup.model_missing"
    # Read from the record rather than the rendered line: the context travels as
    # structured fields, which is what makes it greppable in JSON logs.
    assert record.__dict__["model"] == "gpt-6.1-sun"
    assert record.__dict__["closest"].startswith("gpt-6.1-sol")


async def test_a_served_model_is_not_warned_about(caplog: pytest.LogCaptureFixture) -> None:
    """The ordinary case has to be silent, or the warning is noise."""
    provider = Catalogue(("gpt-6.1-sol",))

    with caplog.at_level(logging.WARNING):
        await _warn_if_model_is_missing(provider, settings_for("gpt-6.1-sol"))  # type: ignore[arg-type]

    assert caplog.text == ""


async def test_a_catalogue_that_cannot_be_read_is_not_a_verdict(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A refused listing says nothing about the model, so it must not warn.

    Treating "the vendor was briefly down" as "your model is gone" would put a
    warning on every deployment of a provider that published no catalogue at all,
    and the first thing an operator learns to do with a warning like that is
    ignore it.
    """
    provider = Catalogue(None)

    with caplog.at_level(logging.WARNING):
        await _warn_if_model_is_missing(provider, settings_for("gpt-6.1-sol"))  # type: ignore[arg-type]

    assert caplog.text == ""
