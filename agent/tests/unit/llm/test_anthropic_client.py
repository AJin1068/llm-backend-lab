"""가짜 전송 계층(httpx2.MockTransport)으로 실제 SDK를 돌린다. 바깥 네트워크는 쓰지 않는다."""

import asyncio
import json
from contextlib import aclosing

import httpx2
import pytest

from taxsplit.core.config import Settings
from taxsplit.llm.anthropic_client import AnthropicLLMClient
from taxsplit.llm.base import (
    CallKind,
    LLMError,
    LLMRequest,
    LLMTimeoutError,
    Message,
    StreamEnd,
    TextDelta,
)
from taxsplit.llm.call_log import InMemoryLlmCallRepository
from taxsplit.llm.recording import RecordingLLMClient

REQUEST = LLMRequest(
    CallKind.MAIN_AGENT,
    [Message("user", "안녕")],
    system="너는 연말정산 도우미다",
    model="claude-sonnet-5-5",
    max_tokens=100,
)

MESSAGE_JSON = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5-5",
    "content": [{"type": "text", "text": "안녕하세요"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 12,
        "output_tokens": 4,
        "cache_creation_input_tokens": 3,
        "cache_read_input_tokens": 5,
    },
}


def sse(*events: dict) -> bytes:
    return "".join(
        f"event: {e['type']}\ndata: {json.dumps(e, ensure_ascii=False)}\n\n" for e in events
    ).encode()


STREAM_EVENTS = [
    {
        "type": "message_start",
        "message": {
            **MESSAGE_JSON,
            "content": [],
            "stop_reason": None,
            "usage": {
                "input_tokens": 12,
                "output_tokens": 1,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "안녕"}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "하세요"}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 4},
    },
    {"type": "message_stop"},
]


class Recorder:
    """가짜 서버. 받은 요청을 모두 적어 둔다."""

    def __init__(self, respond):
        self.requests: list[httpx2.Request] = []
        self._respond = respond

    async def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return await self._respond(request)


def make_client(respond, *, base_url=None, total_timeout=5.0):
    server = Recorder(respond)
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(server))
    client = AnthropicLLMClient(
        api_key="sk-ant-test",
        base_url=base_url,
        total_timeout=total_timeout,
        http_client=http,
    )
    return client, server


async def ok_message(request):
    return httpx2.Response(200, json=MESSAGE_JSON, headers={"request-id": "req_ok"})


async def ok_stream(request):
    return httpx2.Response(
        200,
        content=sse(*STREAM_EVENTS),
        headers={"content-type": "text/event-stream", "request-id": "req_stream"},
    )


def test_complete_converts_response():
    client, server = make_client(ok_message)
    response = asyncio.run(client.complete(REQUEST))

    assert response.text == "안녕하세요"
    assert response.stop_reason == "end_turn"
    assert response.request_id == "req_ok"
    u = response.usage
    assert (u.input_tokens, u.output_tokens) == (12, 4)
    assert (u.cache_creation_input_tokens, u.cache_read_input_tokens) == (3, 5)

    body = json.loads(server.requests[0].content)
    assert body["model"] == "claude-sonnet-5-5"
    assert body["max_tokens"] == 100
    assert body["system"] == "너는 연말정산 도우미다"
    assert body["messages"] == [{"role": "user", "content": "안녕"}]


def test_base_url_is_used():
    client, server = make_client(ok_message, base_url="http://gateway.local:8080")
    asyncio.run(client.complete(REQUEST))

    [request] = server.requests
    assert str(request.url) == "http://gateway.local:8080/v1/messages"


def test_default_base_url_is_anthropic():
    client, server = make_client(ok_message)
    asyncio.run(client.complete(REQUEST))
    assert server.requests[0].url.host == "api.anthropic.com"


@pytest.mark.parametrize("status", [529, 500, 429])
def test_error_sends_exactly_one_http_request(status):
    async def fail(request):
        return httpx2.Response(
            status,
            json={"type": "error", "error": {"type": "overloaded_error", "message": "x"}},
            headers={"request-id": "req_fail"},
        )

    client, server = make_client(fail)
    with pytest.raises(LLMError) as info:
        asyncio.run(client.complete(REQUEST))

    assert len(server.requests) == 1  # SDK 자동 재시도 없음
    assert info.value.status_code == status
    assert info.value.request_id == "req_fail"


def test_http_read_timeout_becomes_llm_timeout_with_one_request():
    async def slow(request):
        raise httpx2.ReadTimeout("느림", request=request)

    client, server = make_client(slow)
    with pytest.raises(LLMTimeoutError):
        asyncio.run(client.complete(REQUEST))
    assert len(server.requests) == 1


def test_total_timeout_becomes_llm_timeout():
    async def hang(request):
        await asyncio.sleep(1)
        return await ok_message(request)

    client, _ = make_client(hang, total_timeout=0.05)
    with pytest.raises(LLMTimeoutError):
        asyncio.run(client.complete(REQUEST))


def test_connection_error_becomes_llm_error():
    async def refuse(request):
        raise httpx2.ConnectError("거절", request=request)

    client, server = make_client(refuse)
    with pytest.raises(LLMError) as info:
        asyncio.run(client.complete(REQUEST))
    assert not isinstance(info.value, LLMTimeoutError)
    assert len(server.requests) == 1


def test_stream_yields_deltas_then_end():
    client, server = make_client(ok_stream)

    async def collect():
        return [e async for e in client.stream(REQUEST)]

    events = asyncio.run(collect())
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["안녕", "하세요"]
    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.response.text == "안녕하세요"
    assert end.response.stop_reason == "end_turn"
    assert end.response.request_id == "req_stream"
    assert (end.response.usage.input_tokens, end.response.usage.output_tokens) == (12, 4)
    assert json.loads(server.requests[0].content)["stream"] is True


def test_stream_error_event_midway_keeps_input_tokens():
    async def broken(request):
        body = sse(
            *STREAM_EVENTS[:3],
            {"type": "error", "error": {"type": "overloaded_error", "message": "과부하"}},
        )
        return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client, server = make_client(broken)
    received = []

    async def consume():
        async for e in client.stream(REQUEST):
            received.append(e)

    with pytest.raises(LLMError) as info:
        asyncio.run(consume())
    assert [e.text for e in received] == ["안녕"]
    assert info.value.usage.input_tokens == 12
    assert len(server.requests) == 1


def test_stream_through_recorder_records_one_with_cost():
    client, _ = make_client(ok_stream)
    repo = InMemoryLlmCallRepository()
    recording = RecordingLLMClient(client, repo, Settings(_env_file=None))

    async def consume_first_then_close():
        async with aclosing(recording.stream(REQUEST)) as events:
            async for _ in events:
                break

    async def consume_all():
        return [e async for e in recording.stream(REQUEST)]

    asyncio.run(consume_all())
    asyncio.run(consume_first_then_close())

    ok, cancelled = repo.items
    assert ok.status == "ok"
    assert ok.provider == "anthropic"
    assert ok.request_id == "req_stream"
    assert ok.cost_usd is not None and ok.cost_usd > 0
    assert cancelled.status == "cancelled"
    assert len(repo.items) == 2
