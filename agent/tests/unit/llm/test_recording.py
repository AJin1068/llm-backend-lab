"""호출 1회 = 기록 1건. 성공·오류·시간 초과·스트리밍 도중 실패·소비자가 중간에 닫음 모두."""

import asyncio
from contextlib import aclosing
from decimal import Decimal

import pytest

from taxsplit.core.config import Settings
from taxsplit.llm.base import (
    CallKind,
    LLMError,
    LLMRequest,
    LLMTimeoutError,
    Message,
    StreamEnd,
    TextDelta,
    Usage,
)
from taxsplit.llm.call_log import InMemoryLlmCallRepository
from taxsplit.llm.fake import FakeLLMClient
from taxsplit.llm.recording import IncompleteStreamError, RecordingLLMClient

REQUEST = LLMRequest(CallKind.MAIN_AGENT, [Message("user", "안녕")])


def make(fake: FakeLLMClient, **settings) -> tuple[RecordingLLMClient, InMemoryLlmCallRepository]:
    repo = InMemoryLlmCallRepository()
    return RecordingLLMClient(fake, repo, Settings(_env_file=None, **settings)), repo


async def drain(client: RecordingLLMClient, request: LLMRequest = REQUEST) -> list:
    return [event async for event in client.stream(request)]


def test_complete_success_records_one():
    client, repo = make(FakeLLMClient("답", usage=Usage(input_tokens=100, output_tokens=20)))
    response = asyncio.run(client.complete(REQUEST))

    assert response.text == "답"
    [call] = repo.items
    assert call.status == "ok"
    assert call.streamed is False
    assert call.kind is CallKind.MAIN_AGENT
    assert (call.input_tokens, call.output_tokens) == (100, 20)
    assert call.stop_reason == "end_turn"
    assert call.latency_ms >= 0


def test_stream_success_records_one():
    client, repo = make(FakeLLMClient(chunks=["가", "나", "다"]))
    events = asyncio.run(drain(client))

    assert [e.text for e in events if isinstance(e, TextDelta)] == ["가", "나", "다"]
    assert isinstance(events[-1], StreamEnd)
    [call] = repo.items
    assert call.status == "ok"
    assert call.streamed is True


def test_complete_api_error_records_one():
    fake = FakeLLMClient(error=LLMError("과부하", status_code=529, request_id="req_1"))
    client, repo = make(fake)
    with pytest.raises(LLMError):
        asyncio.run(client.complete(REQUEST))

    [call] = repo.items
    assert call.status == "error"
    assert call.status_code == 529
    assert call.request_id == "req_1"
    assert call.error_type == "LLMError"


def test_complete_timeout_records_one():
    client, repo = make(FakeLLMClient(error=LLMTimeoutError("시간 초과")))
    with pytest.raises(LLMTimeoutError):
        asyncio.run(client.complete(REQUEST))

    [call] = repo.items
    assert call.status == "timeout"


def test_stream_error_before_first_chunk_records_one():
    client, repo = make(FakeLLMClient(error=LLMTimeoutError("시간 초과")))
    with pytest.raises(LLMTimeoutError):
        asyncio.run(drain(client))

    [call] = repo.items
    assert call.status == "timeout"
    assert call.streamed is True


def test_stream_fails_midway_records_one_with_partial_usage():
    partial = Usage(input_tokens=50)
    fake = FakeLLMClient(
        chunks=["가", "나", "다"], fail_after=2, error=LLMError("끊김", usage=partial)
    )
    client, repo = make(fake)
    received = []

    async def consume():
        async for event in client.stream(REQUEST):
            received.append(event)

    with pytest.raises(LLMError):
        asyncio.run(consume())

    assert len(received) == 2
    [call] = repo.items
    assert call.status == "error"
    assert call.input_tokens == 50


def test_consumer_closes_midway_records_one():
    client, repo = make(FakeLLMClient(chunks=["가", "나", "다"]))

    async def consume_first_only():
        async with aclosing(client.stream(REQUEST)) as events:
            async for _ in events:
                break  # 브라우저를 닫은 것처럼 중간에 끊는다

    asyncio.run(consume_first_only())
    [call] = repo.items
    assert call.status == "cancelled"
    assert call.error_type == "GeneratorExit"


def test_consumer_closes_right_after_last_event_records_one():
    client, repo = make(FakeLLMClient(chunks=["가"]))

    async def consume_until_end():
        async with aclosing(client.stream(REQUEST)) as events:
            async for event in events:
                if isinstance(event, StreamEnd):
                    break  # 마지막 이벤트를 받자마자 닫는다

    asyncio.run(consume_until_end())
    [call] = repo.items
    assert call.status == "ok"


def test_task_cancelled_midway_records_one():
    client, repo = make(FakeLLMClient(chunks=["가", "나", "다"], delay=0.05))

    async def main():
        task = asyncio.create_task(drain(client))
        await asyncio.sleep(0.07)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    [call] = repo.items
    assert call.status == "cancelled"


def test_stream_without_end_event_records_one_error():
    client, repo = make(FakeLLMClient(chunks=["가"], skip_end=True))
    with pytest.raises(IncompleteStreamError):
        asyncio.run(drain(client))

    [call] = repo.items
    assert call.status == "error"
    assert call.error_type == "IncompleteStreamError"


def test_n_calls_make_n_records():
    fake = FakeLLMClient(chunks=["가", "나"])
    client, repo = make(fake)

    async def main():
        await client.complete(REQUEST)
        await drain(client)
        await client.complete(REQUEST)
        async with aclosing(client.stream(REQUEST)) as events:
            async for _ in events:
                break
        fake.error = LLMError("실패")
        with pytest.raises(LLMError):
            await client.complete(REQUEST)

    asyncio.run(main())
    assert [c.status for c in repo.items] == ["ok", "ok", "ok", "cancelled", "error"]
    assert len(fake.requests) == 5


def test_model_and_max_tokens_filled_from_settings_per_kind():
    fake = FakeLLMClient()
    client, repo = make(fake, llm_model_sim_user="claude-haiku-5-5", llm_max_tokens=321)

    async def main():
        await client.complete(LLMRequest(CallKind.SIM_USER, [Message("user", "x")]))
        await client.complete(LLMRequest(CallKind.VERIFIER, [Message("user", "x")]))
        await client.complete(
            LLMRequest(CallKind.VERIFIER, [Message("user", "x")], model="m", max_tokens=7)
        )

    asyncio.run(main())
    assert [(r.model, r.max_tokens) for r in fake.requests] == [
        ("claude-haiku-5-5", 321),
        ("claude-sonnet-5-5", 321),
        ("m", 7),  # 요청에 직접 적은 값이 우선
    ]
    assert [c.kind for c in repo.items] == [CallKind.SIM_USER, CallKind.VERIFIER, CallKind.VERIFIER]


def test_cost_recorded_for_known_model_and_none_for_unknown():
    fake = FakeLLMClient(usage=Usage(input_tokens=1_000_000, output_tokens=0))
    fake.provider = "anthropic"  # 단가표 조회만 실제 공급자처럼
    client, repo = make(fake)

    async def main():
        await client.complete(REQUEST)  # 기본 모델 claude-sonnet-5-5
        await client.complete(LLMRequest(CallKind.MAIN_AGENT, [Message("user", "x")], model="?"))

    asyncio.run(main())
    assert [c.cost_usd for c in repo.items] == [Decimal("2"), None]


def test_fake_provider_cost_is_none():
    client, repo = make(FakeLLMClient())
    asyncio.run(client.complete(REQUEST))
    assert repo.items[0].cost_usd is None


def test_log_line_has_no_api_key_or_prompt(caplog):
    client, _ = make(FakeLLMClient("비밀 답"), anthropic_api_key="sk-ant-secret")
    with caplog.at_level("INFO", logger="taxsplit.llm.calls"):
        asyncio.run(client.complete(LLMRequest(CallKind.MAIN_AGENT, [Message("user", "내 연봉")])))

    [record] = caplog.records
    line = record.getMessage()
    assert '"status": "ok"' in line
    assert "sk-ant-secret" not in line
    assert "내 연봉" not in line
    assert "비밀 답" not in line
