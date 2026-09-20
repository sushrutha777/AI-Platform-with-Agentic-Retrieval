"""Keep the repository test suite independent of live external services."""

import sys
from types import SimpleNamespace


if "litellm" not in sys.modules:
    async def _fake_acompletion(**kwargs):
        if kwargs.get("stream"):
            async def chunks():
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="mock model response")
                        )
                    ]
                )

            return chunks()

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="mock rewritten query")
                )
            ]
        )

    sys.modules["litellm"] = SimpleNamespace(
        suppress_debug_info=False,
        acompletion=_fake_acompletion,
    )

import pytest

from app.guardrails import model_armor


@pytest.fixture(autouse=True)
def disable_live_model_armor(monkeypatch):
    """Disable only the process-level singleton used by application tests."""
    monkeypatch.setattr(model_armor.guardrail, "enabled", False)
