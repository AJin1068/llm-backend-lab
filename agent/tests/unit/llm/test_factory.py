import asyncio

from taxsplit.core.config import Settings
from taxsplit.llm.base import CallKind, LLMRequest, Message
from taxsplit.llm.call_log import InMemoryLlmCallRepository
from taxsplit.llm.factory import build_llm_client


def test_default_settings_build_fake_client_that_records():
    repo = InMemoryLlmCallRepository()
    client = build_llm_client(Settings(_env_file=None), repo)

    asyncio.run(client.complete(LLMRequest(CallKind.SIM_USER, [Message("user", "hi")])))

    assert len(repo.items) == 1
    assert repo.items[0].provider == "fake"
    assert repo.items[0].kind == CallKind.SIM_USER
