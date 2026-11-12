import asyncio
from decimal import Decimal

import pytest

from taxsplit.llm.base import (
    CallKind,
    LLMError,
    LLMRequest,
    LLMTimeoutError,
    Message,
    StreamDone,
    TextDelta,
    Usage,
)
from taxsplit.llm.call_log import InMemoryLlmCallRepository
from taxsplit.llm.fake import FakeLLMClient
from taxsplit.llm.pricing import PriceTable
from taxsplit.llm.recording import RecordingLLMClient

MODELS = {CallKind.MAIN_AGENT: "claude-sonnet-5-5", CallKind.PDF_EXTRACT: "claude-haiku-5-5"}


def make(fake: FakeLLMClient, provider: str = "anthropic"):
    repo = InMemoryLlmCallRepository()
    client = RecordingLLMClient(
        inner=fake,
        provider=provider,
        repo=repo,
        prices=PriceTable.load(),
        model_for=lambda kind: MODELS.get(kind, "claude-sonnet-5-5"),
        default_max_tokens=1000,
    )
    return client, repo


def req(kind: CallKind = CallKind.MAIN_AGENT) -> LLMRequest:
    return LLMRequest(kind=kind, messages=[Message("user", "안녕")])


async def collect(client, request):
    return [e async for e in client.stream(request)]


def test_complete_records_one_call_with_cost():
    fake = FakeLLMClient(usage=Usage(input_tokens=1000, output_tokens=500))
    client, repo = make(fake)

    resp = asyncio.run(client.complete(req()))

    assert resp.text == "fake response"
    assert len(repo.items) == 1
    call = repo.items[0]
    assert (call.kind, call.model, call.status, call.streamed) == (
        CallKind.MAIN_AGENT,
        "claude-sonnet-5-5",
        "ok",
        False,
    )
    assert (call.input_tokens, call.output_tokens) == (1000, 500)
    assert call.cost_usd == Decimal("0.007")  # 1000*2/1e6 + 500*10/1e6


def test_n_calls_make_n_records():
    client, repo = make(FakeLLMClient())

    async def run():
        for _ in range(3):
            await client.complete(req())
        for _ in range(2):
            await collect(client, req())

    asyncio.run(run())
    assert len(repo.items) == 5
    assert [c.streamed for c in repo.items] == [False, False, False, True, True]


def test_model_and_max_tokens_come_from_settings_per_kind():
    fake = FakeLLMClient()
    client, repo = make(fake)

    asyncio.run(client.complete(req(CallKind.PDF_EXTRACT)))
    asyncio.run(client.complete(LLMRequest(CallKind.PDF_EXTRACT, [], model="x", max_tokens=7)))

    assert (fake.requests[0].model, fake.requests[0].max_tokens) == ("claude-haiku-5-5", 1000)
    assert (fake.requests[1].model, fake.requests[1].max_tokens) == ("x", 7)
    assert repo.items[0].kind == CallKind.PDF_EXTRACT


@pytest.mark.parametrize(
    ("error", "status"),
    [(LLMError("boom", status_code=529), "error"), (LLMTimeoutError("slow"), "timeout")],
)
def test_failed_complete_records_one_call_and_reraises(error, status):
    client, repo = make(FakeLLMClient(error=error))

    with pytest.raises(type(error)):
        asyncio.run(client.complete(req()))

    assert len(repo.items) == 1
    call = repo.items[0]
    assert call.status == status
    assert call.error_type == type(error).__name__
    assert call.status_code == error.status_code
    assert call.cost_usd is None


def test_stream_yields_text_then_done_and_records_once():
    client, repo = make(FakeLLMClient(text="공제는 아내 쪽"))

    events = asyncio.run(collect(client, req()))

    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text == "공제는 아내 쪽"
    assert isinstance(events[-1], StreamDone)
    assert len(repo.items) == 1
    assert repo.items[0].status == "ok"
    assert repo.items[0].streamed is True


def test_stream_failing_midway_records_one_error():
    fake = FakeLLMClient(text="a b c d", error=LLMError("cut"), fail_after_chunks=2)
    client, repo = make(fake)

    with pytest.raises(LLMError):
        asyncio.run(collect(client, req()))

    assert [c.status for c in repo.items] == ["error"]


def test_consumer_closing_stream_early_records_one_cancelled():
    client, repo = make(FakeLLMClient(text="a b c d"))

    async def run():
        gen = client.stream(req())
        await anext(gen)
        await gen.aclose()

    asyncio.run(run())
    assert [c.status for c in repo.items] == ["cancelled"]


def test_closing_right_after_done_still_one_ok_record():
    client, repo = make(FakeLLMClient(text="a"))

    async def run():
        gen = client.stream(req())
        async for event in gen:
            if isinstance(event, StreamDone):
                await gen.aclose()
                break

    asyncio.run(run())
    assert [c.status for c in repo.items] == ["ok"]


def test_stream_without_done_is_an_error():
    class NoDone(FakeLLMClient):
        async def stream(self, request):
            yield TextDelta("x")

    client, repo = make(NoDone())

    with pytest.raises(LLMError):
        asyncio.run(collect(client, req()))
    assert [c.status for c in repo.items] == ["error"]


def test_fake_provider_has_no_cost():
    client, repo = make(FakeLLMClient(), provider="fake")
    asyncio.run(client.complete(req()))
    assert repo.items[0].cost_usd is None
