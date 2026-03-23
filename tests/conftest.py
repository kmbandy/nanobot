"""Shared pytest fixtures for nanobot tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse


def make_mock_provider(responses: list[LLMResponse] | None = None) -> MagicMock:
    """Build a MagicMock LLMProvider wired for AgentLoop use.

    Provides the minimal attributes AgentLoop.__init__ reads from the provider,
    plus a pre-configured AsyncMock for chat_with_retry.

    Args:
        responses: Optional list of LLMResponse objects returned in sequence.
            Defaults to a single empty response so the loop terminates cleanly.
    """
    if responses is None:
        responses = [LLMResponse(content="ok", tool_calls=[])]

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation.max_tokens = 4096
    provider.generation.temperature = 0.7
    provider.generation.reasoning_effort = None

    calls = iter(responses)
    provider.chat_with_retry = AsyncMock(side_effect=lambda *a, **kw: next(calls))
    return provider


def make_agent_loop(tmp_path: Path, responses: list[LLMResponse] | None = None) -> AgentLoop:
    """Create a fully wired AgentLoop backed by a mock provider.

    Args:
        tmp_path:  pytest tmp_path fixture value (workspace directory).
        responses: LLMResponse sequence forwarded to make_mock_provider.
    """
    provider = make_mock_provider(responses)
    return AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
    )


@pytest.fixture
def mock_provider() -> MagicMock:
    """Pytest fixture: mock provider with a single default response."""
    return make_mock_provider()


@pytest.fixture
def agent_loop(tmp_path: Path) -> AgentLoop:
    """Pytest fixture: AgentLoop with mock provider, single default response."""
    return make_agent_loop(tmp_path)
