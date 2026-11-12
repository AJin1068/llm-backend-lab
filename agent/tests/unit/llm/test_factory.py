import pytest

from taxsplit.core.config import Settings
from taxsplit.llm.anthropic_client import AnthropicLLMClient
from taxsplit.llm.call_log import InMemoryLlmCallRepository
from taxsplit.llm.factory import build_llm_client, build_provider_client
from taxsplit.llm.fake import FakeLLMClient
from taxsplit.llm.recording import RecordingLLMClient


def test_default_provider_is_fake():
    client = build_llm_client(
        Settings(_env_file=None, llm_provider="fake"), InMemoryLlmCallRepository()
    )
    assert isinstance(client, RecordingLLMClient)
    assert client.provider == "fake"
    assert isinstance(build_provider_client(Settings(_env_file=None)), FakeLLMClient)


def test_anthropic_requires_key():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_provider_client(
            Settings(_env_file=None, llm_provider="anthropic", anthropic_api_key=None)
        )


def test_anthropic_provider_built_with_key():
    settings = Settings(_env_file=None, llm_provider="anthropic", anthropic_api_key="sk-ant-test")
    assert isinstance(build_provider_client(settings), AnthropicLLMClient)
